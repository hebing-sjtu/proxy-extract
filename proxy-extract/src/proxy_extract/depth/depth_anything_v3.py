"""Depth Anything 3 (nested giant+large) as a metric depth backend.

Chosen over mapanything for one operational reason: DA3 carries its DINOv2
backbone inside the single `model.safetensors` it publishes on the Hub, so a
node that can reach Hugging Face can obtain every weight it needs. mapanything
pulls its backbone through `torch.hub` from `dl.fbaipublicfiles.com`, which is
outside most clusters' egress allowlist, and the failure is a silent wait
rather than an error.

Two of the checkpoints are worth telling apart. `DA3METRIC-LARGE` is
Apache-2.0 and small, but it is a DinoV2+DPT pair with no camera head: it
emits *canonical* depth, never sets `is_metric`, and cannot tell you the focal
length you would need to convert it, so it is only metric if some other part
of the system supplies intrinsics. The nested checkpoint below runs the
any-view branch against the metric branch, converts with its own predicted
focal, and reports `is_metric = 1`. It is the one that satisfies delivery's
refusal to encode up-to-scale depth, at the cost of a CC BY-NC 4.0 licence and
6.8 GB of weights.
"""

from __future__ import annotations

import sys
import types

import numpy as np

from ..accel import pick_device, resolve_dtype
from ..cameras import CameraTrack
from .base import DepthResult

# Metric out of the box; CC BY-NC 4.0. Non-commercial use only.
NESTED_CHECKPOINT = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# Apache-2.0, but canonical depth only. Kept as a name rather than a default so
# that choosing it is a deliberate act; delivery will refuse its output unless
# something upstream has made it metric.
METRIC_CHECKPOINT = "depth-anything/DA3METRIC-LARGE"

# `depth_anything_3.api` imports its exporters at module scope, and the model
# graph reaches its multi-view pose aligner while building the Gaussian adapter.
# Between them those drag in gsplat, open3d, pycolmap, moviepy and evo — none of
# which this backend can reach, and none of which it needs: nothing here
# exports, the Gaussian branch stays off, and the aligner returns immediately
# when no input extrinsics are given, which is always our case. Standing in for
# them keeps the install to `--no-deps` plus four small packages.
_OPTIONAL_SUBMODULES = (
    "depth_anything_3.utils.export",
    "depth_anything_3.utils.pose_align",
)


def _make_stub(name: str) -> types.ModuleType:
    """A module whose every attribute is a callable that refuses to run.

    Naming the attributes individually would be tidier, but it also means a new
    symbol upstream — `batch_align_poses_umeyama` was one — turns into an
    ImportError at model construction. Since none of these may legitimately run,
    answering any name and failing only on call is both safer and stabler.
    """
    module = types.ModuleType(name)

    def __getattr__(attribute: str):
        # Introspection must still see a normal module. `inspect` asks for
        # `__file__` while building tracebacks, and handing it a function
        # produces a failure far away from here that looks nothing like a
        # missing dependency.
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)

        def refuse(*_args, **_kwargs):
            raise RuntimeError(
                f"{name}.{attribute} was called, but this environment installed "
                "depth-anything-3 without its optional dependencies. This backend "
                "never exports and never aligns to input poses, so reaching here "
                "means the model was asked for something it was not set up to do."
            )

        return refuse

    module.__getattr__ = __getattr__
    return module


def _stub_unreachable_submodules() -> list[str]:
    """Register stand-ins for submodules whose dependencies are missing.

    Only for those that genuinely fail to import, so a fully installed
    environment keeps the real modules.
    """
    import importlib

    stubbed = []
    for name in _OPTIONAL_SUBMODULES:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = _make_stub(name)
            stubbed.append(name)
    return stubbed


class DepthAnythingV3Backend:
    name = "depth_anything_v3"

    def __init__(
        self,
        *,
        checkpoint: str = NESTED_CHECKPOINT,
        device: str | None = None,
        process_res: int = 504,
        window: int = 1,
        dtype: str = "auto",
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.checkpoint = checkpoint
        self.device = device
        self.process_res = process_res
        self.window = window
        self.dtype = dtype
        self._model = None
        self._stubbed: list[str] = []
        self._resolved_dtype: str | None = None

    def _load(self):
        if self._model is None:
            import torch

            self._stubbed = _stub_unreachable_submodules()
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError as exc:  # pragma: no cover - environment specific
                raise ImportError(
                    "depth-anything-3 is not installed. It is not on PyPI, and its "
                    "declared pins (numpy<2, python<=3.13) fight this environment, so "
                    "install it without them:\n"
                    "  pip install --no-deps --ignore-requires-python "
                    "git+https://github.com/ByteDance-Seed/depth-anything-3\n"
                    "  pip install einops omegaconf addict imageio"
                ) from exc

            self.device = pick_device(self.device)
            self._resolved_dtype = resolve_dtype(self.dtype, self.device)
            model = DepthAnything3.from_pretrained(self.checkpoint)
            model = model.to(getattr(torch, self._resolved_dtype)).to(self.device).eval()
            self._model = model
        return self._model

    def estimate(self, frames: list[np.ndarray], *, cameras: CameraTrack | None = None) -> DepthResult:
        import cv2
        import torch

        model = self._load()
        height, width = frames[0].shape[:2]

        depth_maps: list[np.ndarray] = []
        sky_masks: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        metric = True

        for start in range(0, len(frames), self.window):
            group = frames[start : start + self.window]
            with torch.no_grad():
                prediction = model.inference(list(group), process_res=self.process_res)

            # `is_metric` is an int on the nested model and an empty dict on the
            # plain metric one, so compare rather than trust truthiness.
            metric = metric and prediction.is_metric == 1

            depth = np.asarray(prediction.depth, dtype=np.float32)
            sky = None if prediction.sky is None else np.asarray(prediction.sky)
            confidence = None if prediction.conf is None else np.asarray(prediction.conf, dtype=np.float32)

            for index in range(len(group)):
                depth_maps.append(
                    cv2.resize(depth[index], (width, height), interpolation=cv2.INTER_LINEAR)
                )
                if sky is not None:
                    sky_masks.append(
                        cv2.resize(
                            sky[index].astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    )
                if confidence is not None:
                    confidences.append(
                        cv2.resize(confidence[index], (width, height), interpolation=cv2.INTER_LINEAR)
                    )

        # DA3 fills sky with a finite stand-in (200 m) rather than leaving it
        # undefined. Passing that through would deliver a solid ceiling at 200 m;
        # marking it invalid instead lets the encoders reach for their sky
        # sentinel, which is what both depth.mp4 and duv.mp4 expect.
        valid = ~np.stack(sky_masks) if sky_masks else None

        return DepthResult(
            depth=np.stack(depth_maps),
            metric=metric,
            confidence=np.stack(confidences) if confidences else None,
            valid=valid,
            meta={
                "backend": self.name,
                "checkpoint": self.checkpoint,
                "process_res": self.process_res,
                "window": self.window,
                "device": self.device,
                "dtype": self._resolved_dtype,
                "sky_from_backend": bool(sky_masks),
                "stubbed_submodules": self._stubbed,
                "single_view": self.window == 1,
            },
        )
