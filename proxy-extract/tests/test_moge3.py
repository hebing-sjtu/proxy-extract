"""The MoGe-3 backend, exercised against a stand-in for the real package.

Nothing here downloads a checkpoint, and nothing here could: MoGe-3 depends on
FlexGEMM, which builds on Triton, which publishes no macOS wheels. So what
these cover is the layer this repo actually wrote, which is where the mistakes
would be -- and the mistakes worth catching are not "does it return an array".

The reason this backend exists is that the previous one flickered, and the two
things that make a monocular metric model flicker are both *global* per frame:
the field of view it infers and the metric scale it picks. A test suite that
only checks shapes and dtypes would pass just as happily on a backend that
re-decides both every frame, which is the bug. So the load-bearing tests here
are the ones that pin the locks:

    test_every_frame_of_the_clip_gets_the_same_camera
    test_one_odd_frame_does_not_move_the_locked_camera
    test_the_scale_lock_flattens_the_metric_heads_breathing
    test_the_scale_lock_leaves_the_clips_metric_level_alone

The last of those is the one to be most careful about deleting. Removing scale
jitter and renormalising each clip look the same in every shape check, and the
second destroys the corpus (PROXY_DUV_SPEC.md section 2).
"""

from __future__ import annotations

import contextlib
import functools
import math
import sys
import types

import numpy as np
import pytest

from proxy_extract.depth import get_backend
from proxy_extract.depth.moge3 import (
    CHECKPOINT,
    LARGE_CHECKPOINT,
    MAX_FOV_DEGREES,
    MIN_FOV_DEGREES,
    MoGe3Backend,
    fov_x_from_intrinsics,
)

H, W = 24, 40


class _StandInTensor:
    """Enough of a tensor for the four calls the backend makes on one.

    The backend's preprocessing is `from_numpy -> to(dtype) -> div(255) ->
    permute(2, 0, 1) -> to(device)`, and all of that is numpy underneath. It is
    worth standing in for rather than skipping, because the permute is the step
    that decides whether the model receives CHW or HWC, and getting it wrong is
    a silent transpose that only shows up as bad depth on a real checkpoint.
    """

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def to(self, target):
        if isinstance(target, str):  # a device
            return self
        return _StandInTensor(self._array.astype(target))

    def div(self, value):
        return _StandInTensor(self._array / value)

    def permute(self, *order):
        return _StandInTensor(self._array.transpose(order))

    def __getitem__(self, key):
        return self._array[key]

    @property
    def shape(self):
        return self._array.shape


@pytest.fixture(autouse=True)
def _torch(monkeypatch):
    """Real torch where it exists, a stand-in where it does not.

    MoGe-3 cannot run on this developer platform at all -- no Triton wheels --
    so requiring torch here would mean the locks that are the entire reason for
    this backend are only ever checked on the cluster. The stand-in covers the
    dtype, the two context managers and the array arithmetic, none of which is
    what these tests are about.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        fake = types.ModuleType("torch")
        fake.float32 = np.float32
        fake.bfloat16 = np.float32  # no numpy equivalent, and never asked for here
        fake.from_numpy = _StandInTensor
        fake.no_grad = contextlib.nullcontext
        fake.inference_mode = contextlib.nullcontext
        fake.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", fake)


def _normalised_intrinsics(fov_degrees: float) -> np.ndarray:
    """What MoGe hands back: focal length in units of image width, not pixels."""
    fx = 1.0 / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return np.array([[fx, 0.0, 0.5], [0.0, fx, 0.5], [0.0, 0.0, 1.0]])


class _FakeMoGe:
    """A MoGe-3 stand-in whose per-frame answers the test controls.

    Each frame carries its own index in pixel (0, 0, 0) so this can answer per
    frame without depending on the order it is called in -- which matters,
    because the probe pass and the main pass both come through here and the
    point of several of these tests is *which* frames the probe touched.
    """

    def __init__(
        self,
        *,
        fov_degrees: float | list[float] = 60.0,
        scales: list[float] | None = None,
        invalid_rows: int = 0,
        nan_row: int | None = None,
    ) -> None:
        self.fov_degrees = fov_degrees
        self.scales = scales
        self.invalid_rows = invalid_rows
        self.nan_row = nan_row
        # (frame index, fov_x it was called with) for every forward pass.
        self.calls: list[tuple[int, float | None]] = []
        self.casts: list = []

    # torch plumbing -------------------------------------------------------

    def to(self, *args, **_kwargs):
        self.casts.extend(args)
        return self

    def eval(self):
        return self

    # the model ------------------------------------------------------------

    def _fov_for(self, index: int) -> float:
        if isinstance(self.fov_degrees, list):
            return self.fov_degrees[index]
        return self.fov_degrees

    def infer(
        self,
        image,
        fov_x=None,
        num_tokens=None,
        resolution_level=9,
        refine_steps=3,
        **_kwargs,
    ):
        index = int(round(float(image[0, 0, 0]) * 255.0))
        self.calls.append((index, fov_x))

        # A ramp from 5 m at the top to 15 m at the bottom, times whatever the
        # metric head is doing to the scale of this particular frame.
        column = np.linspace(5.0, 15.0, H, dtype=np.float32)[:, None]
        depth = np.repeat(column, W, axis=1)
        if self.scales is not None:
            depth = depth * self.scales[index]

        mask = np.ones((H, W), dtype=bool)
        if self.invalid_rows:
            mask[: self.invalid_rows, :] = False
            # MoGe leaves the numbers under an invalid mask undefined rather
            # than zero, and a backend that trusted them would deliver a wall.
            depth[: self.invalid_rows, :] = 1e4
        if self.nan_row is not None:
            depth[self.nan_row, :] = np.inf

        return {
            "depth": depth,
            "mask": mask,
            "intrinsics": _normalised_intrinsics(self._fov_for(index)),
        }


class _FakeMoGeV2(_FakeMoGe):
    """A checkpoint with no sparse refinement, i.e. the wrong one.

    `infer` is introspected for its keywords, so the absence of `refine_steps`
    here is the whole point: it is what a v1/v2 checkpoint loaded through the
    v3 class looks like from the adaptation layer's side.
    """

    def infer(self, image, fov_x=None, num_tokens=None, resolution_level=9):
        return super().infer(
            image, fov_x=fov_x, num_tokens=num_tokens, resolution_level=resolution_level
        )


@pytest.fixture
def fake_moge(monkeypatch):
    """Install a fake `moge.model.v3` for the duration of one test."""

    def install(model):
        package = types.ModuleType("moge")
        model_pkg = types.ModuleType("moge.model")
        v3 = types.ModuleType("moge.model.v3")

        class MoGeModel:
            @staticmethod
            def from_pretrained(_checkpoint):
                return model

        v3.MoGeModel = MoGeModel
        for name, module in (
            ("moge", package),
            ("moge.model", model_pkg),
            ("moge.model.v3", v3),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return model

    return install


def _frames(count=8):
    """Static, textured frames, each stamped with its own index.

    Static because the flow used by the scale lock should have nothing to find:
    these tests are about jitter the model invented, so the scene must not be
    supplying real motion that the lock is entitled to keep.
    """
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    frames = []
    for index in range(count):
        frame = texture.copy()
        frame[0, 0, 0] = index
        frames.append(frame)
    return frames


def _per_frame_level(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.array([np.median(d[v]) for d, v in zip(depth, valid)])


# ------------------------------------------------------------------ wiring


def test_the_backend_is_reachable_by_name():
    backend = get_backend("moge3", refine_steps=7)
    assert isinstance(backend, MoGe3Backend)
    assert backend.refine_steps == 7


def test_the_default_checkpoint_is_the_one_that_fits_beside_a_segmenter():
    # vitg is the better model and the wrong default: at six workers per GPU it
    # is the 370M one that actually has room next to a segmentation model.
    assert MoGe3Backend().checkpoint == CHECKPOINT
    assert CHECKPOINT != LARGE_CHECKPOINT


def test_the_options_are_reachable_from_the_command_line():
    # Locking the camera is a constructor keyword, so with no way to pass one
    # through it would be a mode nobody outside this file could turn on.
    from proxy_extract.cli import parse_backend_options

    options = parse_backend_options(["fov_x=55.0", "refine_steps=0"])

    assert options == {"fov_x": 55.0, "refine_steps": 0}
    backend = get_backend("moge3", **options)
    assert backend.fov_x == 55.0
    assert backend.refine_steps == 0


# --------------------------------------------------------------- the camera


@pytest.mark.parametrize("degrees", [30.0, 60.0, 90.0, 120.0])
def test_normalised_intrinsics_round_trip_through_the_fov_helper(degrees):
    recovered = fov_x_from_intrinsics(_normalised_intrinsics(degrees))
    assert recovered == pytest.approx(degrees, abs=1e-9)


def test_a_pixel_focal_length_read_as_normalised_does_not_pass_quietly(fake_moge):
    """The one place a unit mix-up would survive, so it is made to fail.

    A pixel focal length is ~500 where a normalised one is ~1, and reading the
    first as the second gives a FOV of a tenth of a degree. Every frame of the
    clip would then be encoded at a focal length no camera has, and the depth
    would be uniformly wrong by a large factor while looking perfectly stable.
    """
    fov = fov_x_from_intrinsics(np.array([[500.0, 0, 0.5], [0, 500.0, 0.5], [0, 0, 1]]))

    assert fov < MIN_FOV_DEGREES
    fake_moge(_FakeMoGe(fov_degrees=fov))
    with pytest.raises(ValueError, match="fov_x="):
        MoGe3Backend(device="cpu").estimate(_frames(4))


def test_a_non_positive_focal_length_is_refused():
    with pytest.raises(ValueError, match="focal length"):
        fov_x_from_intrinsics(np.zeros((3, 3)))


def test_every_frame_of_the_clip_gets_the_same_camera(fake_moge):
    """The first half of the anti-flicker claim.

    MoGe re-infers the camera per frame, and metric depth comes out through the
    focal length, so a degree of FOV wobble moves every pixel of the frame at
    once. That is the component a per-pixel temporal median cannot touch. So
    the camera is solved once and handed to every frame.
    """
    # A camera that wobbles by several degrees over the clip, as a real one's
    # inferred value does.
    wobble = [58.0, 61.0, 59.5, 62.0, 57.5, 60.5, 59.0, 61.5]
    model = fake_moge(_FakeMoGe(fov_degrees=wobble))

    result = MoGe3Backend(device="cpu", lock_scale=False).estimate(_frames(8))

    main_pass = [fov for _index, fov in model.calls if fov is not None]
    assert len(main_pass) == 8, "every frame should have been given a camera"
    assert len(set(main_pass)) == 1, "the camera must not be re-decided per frame"
    assert main_pass[0] == pytest.approx(result.meta["fov_x_degrees"], abs=1e-4)
    # And the report says how much flicker the lock took out.
    assert result.meta["fov_probe_spread_degrees"] == pytest.approx(4.5, abs=1e-3)


def test_one_odd_frame_does_not_move_the_locked_camera(fake_moge):
    """Median, not mean: a single close-up would otherwise spoil all 124 frames.

    If one probe frame reads as a close-up its focal length is an outlier, and
    under a mean that one frame's mistake is spread over the whole clip -- a
    stable answer that is stable at the wrong value, which is worse than the
    flicker it replaced because nothing downstream can see it.
    """
    outlier = [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 150.0]
    fake_moge(_FakeMoGe(fov_degrees=outlier))

    locked = MoGe3Backend(device="cpu", lock_scale=False).estimate(
        _frames(8)
    ).meta["fov_x_degrees"]

    assert locked == pytest.approx(60.0, abs=1e-3)
    assert locked < np.mean(outlier), "a mean would have been dragged upward"


def test_the_probe_is_spread_over_the_clip_not_taken_from_the_front(fake_moge):
    """An episode that starts indoors and ends on a street has two framings.

    A probe drawn from the first few frames measures the corridor's camera and
    applies it to the street, so the sample has to span the clip.
    """
    model = fake_moge(_FakeMoGe())

    MoGe3Backend(device="cpu", fov_probe_frames=4, lock_scale=False).estimate(
        _frames(20)
    )

    probed = [index for index, fov in model.calls if fov is None]
    assert len(probed) == 4
    assert probed == sorted(probed)
    assert probed[0] == 0
    assert probed[-1] >= 15, f"the probe never reached the end of the clip: {probed}"


def test_a_known_camera_skips_the_probe_entirely(fake_moge):
    """Given intrinsics are strictly better than probed ones, and cheaper.

    COLMAP's pixel focal length is the one quantity a sparse reconstruction
    gets right despite being defined only up to a similarity, so when it is
    available there is nothing to estimate and eight forward passes to save.
    """
    model = fake_moge(_FakeMoGe(fov_degrees=91.0))

    result = MoGe3Backend(device="cpu", fov_x=55.0, lock_scale=False).estimate(
        _frames(6)
    )

    assert [fov for _index, fov in model.calls] == [55.0] * 6
    assert result.meta["fov_source"] == "given"
    assert result.meta["fov_probe_frames"] == 0
    assert result.meta["fov_x_degrees"] == 55.0


def test_a_collapsed_probe_is_refused_rather_than_delivered(fake_moge):
    fake_moge(_FakeMoGe(fov_degrees=MAX_FOV_DEGREES + 5.0))

    with pytest.raises(ValueError) as caught:
        MoGe3Backend(device="cpu").estimate(_frames(4))

    message = str(caught.value)
    assert "FOV probe" in message
    assert "fov_x=" in message, "the refusal has to say how to override it"


def test_a_camera_outside_any_real_lens_is_refused_before_anything_loads():
    with pytest.raises(ValueError, match="outside"):
        MoGe3Backend(fov_x=MAX_FOV_DEGREES + 1.0)
    with pytest.raises(ValueError, match="outside"):
        MoGe3Backend(fov_x=MIN_FOV_DEGREES - 1.0)


# ----------------------------------------------------------------- the depth


def test_moges_invalid_region_is_not_delivered_as_a_surface(fake_moge):
    """Unlike DA3, MoGe declines to place the sky rather than putting a ceiling
    there -- but the numbers under its mask are undefined, not zero, so reading
    depth without the mask delivers a wall at whatever the head happened to
    emit. Here that is 10 km.
    """
    fake_moge(_FakeMoGe(invalid_rows=5))

    result = MoGe3Backend(device="cpu", lock_scale=False).estimate(_frames(4))

    valid = result.valid_mask()
    assert not valid[:, :5, :].any(), "the sky should be invalid"
    assert valid[:, 5:, :].all(), "the ground should survive"
    # What delivery encodes: invalid becomes the 0 sentinel, and the 10 km
    # garbage never reaches the file.
    assert result.depth[:, :5, :].max() == 0.0
    assert result.depth.max() < 20.0


def test_non_finite_depth_inside_the_mask_is_dropped(fake_moge):
    """A horizon pixel can come back infinite while the mask still claims it.

    Both conditions are kept for this reason; an inf would otherwise survive
    into the float32 plane and take the audit's p50 with it.
    """
    fake_moge(_FakeMoGe(nan_row=H - 1))

    result = MoGe3Backend(device="cpu", lock_scale=False).estimate(_frames(4))

    assert not result.valid_mask()[:, -1, :].any()
    assert np.isfinite(result.depth).all()


def test_the_prediction_is_reported_as_metric(fake_moge):
    # Delivery refuses a non-metric stack, so this flag is what makes the
    # backend usable at all rather than a detail of the report.
    fake_moge(_FakeMoGe())
    assert MoGe3Backend(device="cpu").estimate(_frames(4)).metric is True


# ----------------------------------------------------------------- the scale


def test_the_scale_lock_flattens_the_metric_heads_breathing(fake_moge):
    """The second half of the anti-flicker claim.

    A static scene whose depth is multiplied by a different factor every frame
    is exactly the metric head breathing: no pixel moved, but every pixel's
    number did, by the same ratio. A windowed per-pixel median cannot fix this
    because the neighbours it votes against are wrong by the same factor.

    The measurement is the *frame-to-frame* step rather than the spread over the
    clip, and the difference is the whole contract. Flicker is what changes
    between consecutive frames; a slow drift across a clip is what a camera
    dollying back looks like, and the lock is required to leave that alone. So a
    clip-wide max/min would be a test that fails when the lock behaves
    correctly. A pure two-frame oscillation is the worst case for the straight
    line that separates the two, and it does leave a few percent behind as a
    ramp -- see the assertion below, which pins that it is a ramp and not a
    jump.
    """
    jitter = [1.0 + 0.05 * (-1) ** index for index in range(8)]
    fake_moge(_FakeMoGe(scales=jitter))

    loose = MoGe3Backend(device="cpu", lock_scale=False).estimate(_frames(8))
    locked = MoGe3Backend(device="cpu", lock_scale=True).estimate(_frames(8))

    before = _per_frame_level(loose.depth, loose.valid_mask())
    after = _per_frame_level(locked.depth, locked.valid_mask())

    def largest_step(levels):
        return float(np.max(np.abs(np.diff(np.log(levels)))))

    assert before.max() / before.min() == pytest.approx(1.05 / 0.95, rel=1e-3)
    # 10.5% of depth, jumping every frame, is the flicker being complained about.
    assert largest_step(before) == pytest.approx(math.log(1.05 / 0.95), rel=1e-3)
    # Under 1% per frame, and monotonic, so what is left is drift and not flicker.
    assert largest_step(after) < 0.01, f"still breathing: {after}"
    assert np.all(np.diff(after) < 0.0), f"not a smooth trend: {after}"
    assert locked.meta["scale_locked"] is True
    assert locked.meta["scale_jitter_removed_pct"] > 1.0


def test_the_scale_lock_leaves_the_clips_metric_level_alone(fake_moge):
    """The test to be most reluctant to delete.

    Removing jitter and renormalising each clip are indistinguishable in every
    shape, dtype and range check, and the second is the single most destructive
    thing that can happen to this dataset: the depth channel stops being the
    same physical quantity across the corpus while every file stays valid and
    the run trains and scores while learning nothing.

    The guarantee is that the mean log correction over the clip is exactly
    zero, so the clip's geometric mean depth -- where the encoder's log codes
    sit -- comes out as it went in.
    """
    jitter = [1.0 + 0.05 * (-1) ** index for index in range(8)]
    fake_moge(_FakeMoGe(scales=jitter))

    loose = MoGe3Backend(device="cpu", lock_scale=False).estimate(_frames(8))
    locked = MoGe3Backend(device="cpu", lock_scale=True).estimate(_frames(8))

    valid = loose.valid_mask()
    before = float(np.exp(np.mean(np.log(loose.depth[valid]))))
    after = float(np.exp(np.mean(np.log(locked.depth[locked.valid_mask()]))))

    assert after == pytest.approx(before, rel=1e-3)
    assert locked.meta["scale_mean_log_correction"] == pytest.approx(0.0, abs=1e-9)


def test_the_lock_can_be_turned_off_and_says_so(fake_moge):
    fake_moge(_FakeMoGe())
    meta = MoGe3Backend(device="cpu", lock_scale=False).estimate(_frames(8)).meta
    assert meta["scale_locked"] is False


def test_a_clip_too_short_to_lock_is_reported_rather_than_guessed_at(fake_moge):
    """Two frames give one ratio, which is a level curve with no trend to fit.

    The interesting half is that it is reported: a silent pass-through here is
    a clip whose depth is unlocked while the run's report says the lock was on.
    """
    fake_moge(_FakeMoGe(scales=[1.0, 1.1]))

    meta = MoGe3Backend(device="cpu", lock_scale=True).estimate(_frames(2)).meta

    assert meta["scale_locked"] is False


# ---------------------------------------------------------------- the report


def test_the_report_records_the_clip_the_call_actually_saw(fake_moge):
    """Both locks are per call, so a caller that batches splits them.

    `--chunk-frames 32` over a 124-frame window re-probes the camera and
    re-levels the scale four times, and the seam is a step in depth at frames
    32, 64 and 96. Nothing here can prevent that, so it is recorded: this
    number not matching the window length is the explanation for a seam.
    """
    fake_moge(_FakeMoGe())

    meta = MoGe3Backend(device="cpu").estimate(_frames(6)).meta

    assert meta["frames_in_call"] == 6
    assert meta["backend"] == "moge3"
    assert meta["checkpoint"] == CHECKPOINT
    assert meta["fov_source"] == "probed"


# -------------------------------------------------------------- the refusals


def test_a_v2_checkpoint_loaded_through_the_v3_class_is_refused_loudly(fake_moge):
    """Sparse refinement is the reason for this backend, so losing it silently
    is the failure mode that would waste a whole run: the prediction still
    arrives, still looks metric, and is a MoGe-2 prediction.
    """
    fake_moge(_FakeMoGeV2())

    with pytest.raises(ValueError) as caught:
        MoGe3Backend(device="cpu").estimate(_frames(4))

    message = str(caught.value)
    assert "refine_steps" in message
    assert "v1/v2" in message


def test_turning_refinement_off_is_passed_through_not_dropped(fake_moge):
    """`refine_steps=0` is a real setting -- the MoGe-2-class baseline to
    compare against -- and it is falsy, so a truthiness filter would drop it
    and quietly run the default 3.
    """
    model = fake_moge(_FakeMoGe())
    seen: list = []
    original = model.infer

    # `functools.wraps` is not decoration for its own sake here: the backend
    # asks `inspect.signature` which keywords the checkpoint accepts, so a
    # bare `(image, **kwargs)` spy would look like a checkpoint that accepts
    # nothing and be refused before it recorded anything.
    @functools.wraps(original)
    def record(image, **kwargs):
        seen.append(kwargs.get("refine_steps", "absent"))
        return original(image, **kwargs)

    model.infer = record

    MoGe3Backend(device="cpu", refine_steps=0, fov_x=60.0).estimate(_frames(3))

    assert seen == [0, 0, 0]


def test_negative_refinement_is_refused_before_anything_loads():
    with pytest.raises(ValueError, match="refine_steps"):
        MoGe3Backend(refine_steps=-1)


def test_a_probe_of_no_frames_is_refused_before_anything_loads():
    with pytest.raises(ValueError, match="fov_probe_frames"):
        MoGe3Backend(fov_probe_frames=0)


def test_an_empty_clip_is_refused(fake_moge):
    fake_moge(_FakeMoGe())
    with pytest.raises(ValueError, match="at least one frame"):
        MoGe3Backend(device="cpu").estimate([])


def test_a_missing_moge_package_says_macos_cannot_run_it(monkeypatch):
    """The install is a git URL and the platform is a hard no, so the message
    has to carry both or the next person spends an afternoon on pip.
    """
    for name in ("moge", "moge.model", "moge.model.v3"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = __import__

    def refuse(name, *args, **kwargs):
        if name.startswith("moge"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse)

    with pytest.raises(ImportError) as caught:
        MoGe3Backend(device="cpu").estimate(_frames(2))

    message = str(caught.value)
    assert "github.com/microsoft/MoGe" in message
    assert "macOS" in message and "Triton" in message
