"""MoGe-3 as the metric depth backend, with its camera and its scale locked
across the clip rather than re-decided every frame.

Why this backend exists at all is fine detail: MoGe-3's Self-Guided Sparse 3D
Refinement lifts the point map onto a voxel shell and refines it with sparse 3D
convolutions, so features do not mix across a depth discontinuity. That is
exactly the failure that makes a 2D decoder smear a railing or a lamp post into
the wall behind it, and after the reduction to 192x336 a smeared thin structure
is the difference between a pole and nothing.

But swapping the checkpoint is not what fixes flicker, and it is worth being
precise about why. A monocular model run frame by frame re-estimates two global
quantities per frame, and both of them move the whole depth map at once:

1. **The field of view.** Metric depth comes out through the focal length, so a
   degree of FOV wobble is a few percent of depth on every pixel of the frame
   simultaneously. `infer()` takes `fov_x` for exactly this reason, so this
   backend solves the camera once over a handful of probe frames and then hands
   the same value to every frame of the clip.
2. **The metric scale.** Even with the camera pinned, the metric head's output
   is not perfectly stable frame to frame. `temporal.lock_depth_scale` removes
   the high-frequency part of that drift while leaving the clip's absolute
   level alone — see its docstring for why that distinction is the whole game
   and why a per-clip renormalisation would be the worst possible fix.

Neither of these is reachable through the `DepthBackend` protocol from outside:
`estimate()` is handed a whole clip precisely so a backend can do work that
needs the sequence. A caller batching frames through `--chunk-frames` splits the
clip into independent reconstructions and gets a seam at every boundary, so the
report records how many frames this call actually saw.

Two deployment notes:

- **macOS cannot run this.** MoGe-3 depends on FlexGEMM, which builds on Triton,
  and Triton publishes no macOS wheels. The tests here run against a stand-in.
- MoGe-3 requires an explicit checkpoint — unlike v1/v2 there is no default
  inside the package — so `CHECKPOINT` below is this pipeline's choice, not
  MoGe's.
"""

from __future__ import annotations

import inspect
import math

import numpy as np

from ..accel import pick_device
from ..cameras import CameraTrack
from ..temporal import lock_depth_scale
from .base import DepthResult

# The 370M ViT-L. Metric scale and normals, MIT licence, and the one that fits
# beside a segmentation model on a single card. `moge-3-vitg` is 1.25B and
# better; it is a name here rather than the default because at six workers per
# GPU the large one is what actually runs.
CHECKPOINT = "Ruicheng/moge-3-vitl"
LARGE_CHECKPOINT = "Ruicheng/moge-3-vitg"

# Sparse refinement steps. Three is MoGe-3's own default and what the paper
# reports; the model trains at three and generalises to seven at test time.
# Zero turns SSR off entirely and leaves the MoGe-2-class base prediction, which
# is the comparison to make if fine detail is ever suspected of costing more
# than it buys.
DEFAULT_REFINE_STEPS = 3

# Frames sampled to solve the clip's field of view. Spread over the clip rather
# than taken from the front, because a probe drawn from one end measures that
# end's framing: an episode that starts indoors and ends on a street would take
# its camera from a corridor. Eight is where the median stops moving on the
# clips measured here, and each one costs a forward pass that is thrown away.
DEFAULT_FOV_PROBE_FRAMES = 8

# Wider than any real camera, and wide enough that a probe which has collapsed
# produces a refusal rather than a clip whose depth is uniformly wrong by a
# factor. A fisheye episode would legitimately trip this, and should: MoGe's
# own supported range is 2:1 to 1:2 aspect, not fisheye.
MIN_FOV_DEGREES = 10.0
MAX_FOV_DEGREES = 160.0



def fov_x_from_intrinsics(intrinsics) -> float:
    """Horizontal FOV in degrees, from MoGe's *normalised* intrinsics.

    Normalised means `fx` is in units of image width, so `tan(fov_x / 2)` is
    `1 / (2 * fx)` and the pixel dimensions never enter. Reading it as pixel
    focal length instead gives a FOV of a small fraction of a degree, which
    `_solve_fov` would reject rather than quietly pass on — that refusal is
    there because this conversion is the one place a unit mix-up would survive.
    """
    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    fx = float(matrix[0, 0])
    if not math.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"intrinsics have a non-positive focal length: {fx}")
    return math.degrees(2.0 * math.atan(1.0 / (2.0 * fx)))


class MoGe3Backend:
    name = "moge3"

    def __init__(
        self,
        *,
        checkpoint: str = CHECKPOINT,
        device: str | None = None,
        refine_steps: int = DEFAULT_REFINE_STEPS,
        num_tokens: int | None = None,
        resolution_level: int = 9,
        fov_x: float | None = None,
        fov_probe_frames: int = DEFAULT_FOV_PROBE_FRAMES,
        lock_scale: bool = True,
        scale_radius: int = 12,
        scale_flow_downscale: int = 4,
        dtype: str = "float32",
    ) -> None:
        if refine_steps < 0:
            raise ValueError(f"refine_steps must be >= 0, got {refine_steps}")
        if fov_probe_frames < 1:
            raise ValueError(f"fov_probe_frames must be >= 1, got {fov_probe_frames}")
        if fov_x is not None and not MIN_FOV_DEGREES <= fov_x <= MAX_FOV_DEGREES:
            raise ValueError(
                f"fov_x={fov_x} is outside [{MIN_FOV_DEGREES}, {MAX_FOV_DEGREES}] degrees"
            )
        self.checkpoint = checkpoint
        self.device = device
        self.refine_steps = refine_steps
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        # A known camera. Given, the probe pass is skipped entirely, which is
        # both faster and strictly better: ABot's COLMAP model carries a pixel
        # focal length, and pixel focal length is the one quantity a sparse
        # reconstruction gets right despite being defined only up to a
        # similarity. See RUNBOOK section 9.
        self.fov_x = fov_x
        self.fov_probe_frames = fov_probe_frames
        self.lock_scale = lock_scale
        self.scale_radius = scale_radius
        self.scale_flow_downscale = scale_flow_downscale
        self.dtype = dtype
        self._model = None
        self._infer_keywords: frozenset[str] = frozenset()

    # ------------------------------------------------------------------ model

    def _load(self):
        if self._model is None:
            import torch

            try:
                from moge.model.v3 import MoGeModel
            except ImportError as exc:  # pragma: no cover - environment specific
                raise ImportError(
                    "moge is not installed, and it is not on PyPI:\n"
                    "  pip install git+https://github.com/microsoft/MoGe.git\n"
                    "macOS is not supported — MoGe-3's FlexGEMM dependency builds on "
                    "Triton, which publishes no macOS wheels."
                ) from exc

            self.device = pick_device(self.device)
            model = MoGeModel.from_pretrained(self.checkpoint)
            model = model.to(getattr(torch, self.dtype)).to(self.device).eval()
            self._model = model
            # `infer` has gained keywords across v1, v2 and v3 and will gain
            # more. Asking the signature which ones exist means a newer MoGe
            # does not have to be matched here, and an older one refuses a
            # setting loudly instead of ignoring it.
            self._infer_keywords = frozenset(
                inspect.signature(model.infer).parameters
            )
        return self._model

    def _infer(self, frame: np.ndarray, *, fov_x: float | None) -> dict:
        import torch

        model = self._load()
        tensor = torch.from_numpy(np.ascontiguousarray(frame))
        tensor = tensor.to(getattr(torch, self.dtype)).div(255.0).permute(2, 0, 1)

        wanted = {
            "fov_x": fov_x,
            "num_tokens": self.num_tokens,
            "resolution_level": self.resolution_level,
            "refine_steps": self.refine_steps,
        }
        keywords = {
            key: value
            for key, value in wanted.items()
            if value is not None and key in self._infer_keywords
        }
        refused = [
            key
            for key, value in wanted.items()
            if value is not None and key not in self._infer_keywords
        ]
        if refused:
            raise ValueError(
                f"{self.checkpoint} does not accept {', '.join(sorted(refused))}; "
                f"its infer() takes {sorted(self._infer_keywords)}. `refine_steps` "
                "missing means the checkpoint is a v1/v2 model loaded through the v3 "
                "class, in which case use the depth_anything_v3 backend or a v3 "
                "checkpoint rather than silently getting an unrefined prediction."
            )

        with torch.no_grad():
            output = model.infer(tensor.to(self.device), **keywords)
        return {key: _to_numpy(value) for key, value in output.items()}

    # ------------------------------------------------------------------ camera

    def _probe_indices(self, count: int) -> list[int]:
        """Which frames the FOV is solved on: evenly spread, never fewer than one."""
        wanted = min(self.fov_probe_frames, count)
        step = count / wanted
        return sorted({min(int(index * step), count - 1) for index in range(wanted)})

    def _solve_fov(self, frames: list[np.ndarray]) -> tuple[float, dict]:
        """One field of view for the whole clip, as the median of a few probes.

        Median rather than mean, and rather than the first frame's: a single
        frame that MoGe reads as a close-up gives an outlying focal length,
        and with a mean that one frame's mistake is spread over all 124.
        """
        if self.fov_x is not None:
            return self.fov_x, {"fov_source": "given", "fov_probe_frames": 0}

        indices = self._probe_indices(len(frames))
        degrees = [
            fov_x_from_intrinsics(self._infer(frames[index], fov_x=None)["intrinsics"])
            for index in indices
        ]
        median = float(np.median(degrees))
        if not MIN_FOV_DEGREES <= median <= MAX_FOV_DEGREES:
            raise ValueError(
                f"the FOV probe solved {median:.2f} degrees over frames {indices}, "
                f"outside [{MIN_FOV_DEGREES}, {MAX_FOV_DEGREES}]. Every frame of this "
                "clip would then be encoded at a focal length no camera has, so this "
                "refuses rather than delivering it. Pass fov_x= if the camera is known."
            )
        spread = float(np.max(degrees) - np.min(degrees)) if len(degrees) > 1 else 0.0
        return median, {
            "fov_source": "probed",
            "fov_probe_frames": len(indices),
            "fov_probe_indices": indices,
            # How much the per-frame camera moved over the clip, which is the
            # size of the flicker this lock removes. Worth having in the report:
            # a clip where this is near zero did not need the probe, and one
            # where it is several degrees is one where locking mattered.
            "fov_probe_spread_degrees": round(spread, 4),
        }

    # ------------------------------------------------------------------ public

    def estimate(
        self, frames: list[np.ndarray], *, cameras: CameraTrack | None = None
    ) -> DepthResult:
        if not frames:
            raise ValueError("estimate() needs at least one frame")

        fov_x, camera_info = self._solve_fov(frames)

        depth_maps: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        for frame in frames:
            output = self._infer(frame, fov_x=fov_x)
            depth = np.asarray(output["depth"], dtype=np.float32)
            # MoGe marks sky and anything else it declines to place as invalid
            # rather than filling it with a finite stand-in, so the mask is
            # usable directly — unlike DA3, which puts a 200 m ceiling there.
            # Non-finite depth inside the mask is still possible at the horizon,
            # so both conditions are kept.
            mask = np.asarray(output["mask"], dtype=bool) & np.isfinite(depth)
            depth_maps.append(np.where(mask, depth, 0.0).astype(np.float32))
            valid_masks.append(mask)

        stack = np.stack(depth_maps)
        valid = np.stack(valid_masks)

        scale_info: dict = {"scale_locked": False}
        if self.lock_scale and len(frames) >= 3:
            stack, scale_info = lock_depth_scale(
                stack,
                guide_frames=frames,
                radius=self.scale_radius,
                flow_downscale=self.scale_flow_downscale,
            )

        return DepthResult(
            depth=stack,
            metric=True,
            valid=valid,
            meta={
                "backend": self.name,
                "checkpoint": self.checkpoint,
                "device": self.device,
                "dtype": self.dtype,
                "refine_steps": self.refine_steps,
                "resolution_level": self.resolution_level,
                "num_tokens": self.num_tokens,
                "fov_x_degrees": round(float(fov_x), 4),
                **camera_info,
                **scale_info,
                # The clip this call saw. A caller that batches frames splits
                # the clip into independent reconstructions, and both locks are
                # then per batch: the FOV is re-probed and the scale re-levelled
                # at every boundary. Recorded rather than warned about, because
                # for a 124-frame window the batch is the clip and nothing is
                # wrong.
                "frames_in_call": len(frames),
            },
        )


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    return np.asarray(value)
