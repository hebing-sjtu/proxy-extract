"""The DA3 backend, exercised against a stand-in for the real package.

The checkpoint is 6.8 GB, so nothing here downloads it. What these cover is the
adaptation layer, which is where the mistakes would be: a sky that arrives as a
finite 200 m ceiling, a prediction that comes back at the processing
resolution rather than the frame's, and an `is_metric` field that is an int on
one checkpoint and an empty dict on another.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from proxy_extract.depth import get_backend
from proxy_extract.depth.depth_anything_v3 import (
    METRIC_CHECKPOINT,
    NESTED_CHECKPOINT,
    DepthAnythingV3Backend,
)

PROCESSED = (28, 50)  # what DA3 hands back at process_res, not the frame size
FRAME = (72, 128)


class _Prediction:
    def __init__(self, depth, *, is_metric, sky=None, conf=None):
        self.depth = depth
        self.is_metric = is_metric
        self.sky = sky
        self.conf = conf
        self.extrinsics = None
        self.intrinsics = None
        self.scale_factor = None


class _FakeModel:
    """Records how it was called so the tests can assert on grouping."""

    def __init__(self, *, is_metric=1, with_sky=True, sky_depth=200.0):
        self.is_metric = is_metric
        self.with_sky = with_sky
        self.sky_depth = sky_depth
        self.calls: list[int] = []
        self.process_res: list[int] = []

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def inference(self, images, *, process_res=504, **_kwargs):
        self.calls.append(len(images))
        self.process_res.append(process_res)
        count = len(images)
        depth = np.full((count, *PROCESSED), 7.0, dtype=np.float32)
        sky = None
        if self.with_sky:
            sky = np.zeros((count, *PROCESSED), dtype=bool)
            sky[:, :4, :] = True  # a band of sky along the top
            depth[sky] = self.sky_depth
        conf = np.full((count, *PROCESSED), 0.5, dtype=np.float32)
        return _Prediction(depth, is_metric=self.is_metric, sky=sky, conf=conf)


@pytest.fixture(autouse=True)
def _isolate_da3_modules():
    """Keep each test's view of `depth_anything_3` to itself.

    The backend registers its stand-ins in `sys.modules`, so without this the
    first test to load leaves them behind and every later one skips the
    stubbing it meant to exercise.
    """
    before = {name: module for name, module in sys.modules.items() if name.startswith("depth_anything_3")}
    yield
    for name in [name for name in sys.modules if name.startswith("depth_anything_3")]:
        del sys.modules[name]
    sys.modules.update(before)


@pytest.fixture
def fake_da3(monkeypatch):
    """Install a fake `depth_anything_3.api` for the duration of one test."""

    def install(model):
        package = types.ModuleType("depth_anything_3")
        utils = types.ModuleType("depth_anything_3.utils")
        api = types.ModuleType("depth_anything_3.api")

        class DepthAnything3:
            @staticmethod
            def from_pretrained(_checkpoint):
                return model

        api.DepthAnything3 = DepthAnything3
        for name, module in (
            ("depth_anything_3", package),
            ("depth_anything_3.utils", utils),
            ("depth_anything_3.api", api),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return model

    return install


def _frames(count=2):
    return [np.zeros((*FRAME, 3), dtype=np.uint8) for _ in range(count)]


def test_the_backend_is_reachable_by_name():
    backend = get_backend("depth_anything_v3", window=3)
    assert isinstance(backend, DepthAnythingV3Backend)
    assert backend.window == 3


def test_the_default_checkpoint_is_the_one_that_reports_metres():
    # The Apache checkpoint is cheaper and smaller, and it is the wrong default:
    # it never sets is_metric, so delivery would refuse every episode.
    assert DepthAnythingV3Backend().checkpoint == NESTED_CHECKPOINT
    assert NESTED_CHECKPOINT != METRIC_CHECKPOINT


def test_depth_comes_back_at_the_frame_size_not_the_processing_size(fake_da3):
    fake_da3(_FakeModel())
    backend = DepthAnythingV3Backend(device="cpu")

    result = backend.estimate(_frames(2))

    assert result.depth.shape == (2, *FRAME)
    assert result.confidence.shape == (2, *FRAME)


def test_the_nested_checkpoint_is_taken_as_metric(fake_da3):
    fake_da3(_FakeModel(is_metric=1))
    result = DepthAnythingV3Backend(device="cpu").estimate(_frames())
    assert result.metric is True


def test_an_empty_is_metric_is_not_mistaken_for_metres(fake_da3):
    # DA3METRIC-LARGE returns `{}` here. It is falsy, but a truthiness test is
    # not what protects us — a future value of `0` or `"no"` should fail too.
    fake_da3(_FakeModel(is_metric={}))
    result = DepthAnythingV3Backend(device="cpu").estimate(_frames())
    assert result.metric is False


def test_the_two_hundred_metre_sky_does_not_survive_as_a_surface(fake_da3):
    fake_da3(_FakeModel(with_sky=True, sky_depth=200.0))

    result = DepthAnythingV3Backend(device="cpu").estimate(_frames(1))

    # The sky band is the top 4 of 28 processed rows, so it lands somewhere
    # around row 10 once it is scaled up to the frame; assert either side of it.
    band = round(4 / PROCESSED[0] * FRAME[0])
    valid = result.valid_mask()
    assert not valid[0, :band, :].any(), "sky should be invalid, not a ceiling at 200 m"
    assert valid[0, band + 2 :, :].all(), "ground should survive"
    # What delivery actually encodes: invalid becomes 0, which is the sentinel.
    encoded = np.where(valid, result.depth, 0.0)
    assert encoded[0, :4, :].max() == 0.0
    assert 200.0 not in np.unique(encoded)


def test_a_backend_without_a_sky_head_leaves_every_pixel_valid(fake_da3):
    fake_da3(_FakeModel(with_sky=False))

    result = DepthAnythingV3Backend(device="cpu").estimate(_frames(1))

    assert result.valid is None
    assert result.valid_mask().all()
    assert result.meta["sky_from_backend"] is False


def test_frames_go_in_one_at_a_time_by_default(fake_da3):
    model = fake_da3(_FakeModel())

    DepthAnythingV3Backend(device="cpu").estimate(_frames(5))

    assert model.calls == [1, 1, 1, 1, 1]


def test_a_window_groups_frames_into_one_reconstruction(fake_da3):
    # DA3 is an any-view model: several frames in one call are treated as views
    # of a single scene, which ties their scale together.
    model = fake_da3(_FakeModel())

    result = DepthAnythingV3Backend(device="cpu", window=2).estimate(_frames(5))

    assert model.calls == [2, 2, 1]
    assert result.depth.shape[0] == 5
    assert result.meta["single_view"] is False


def test_the_processing_resolution_is_passed_through(fake_da3):
    model = fake_da3(_FakeModel())
    DepthAnythingV3Backend(device="cpu", process_res=728).estimate(_frames(1))
    assert model.process_res == [728]


def test_a_zero_window_is_refused_before_anything_loads():
    with pytest.raises(ValueError, match="window"):
        DepthAnythingV3Backend(window=0)


def test_the_window_is_reachable_from_the_command_line():
    # It is a constructor keyword, so without a way to pass one through it would
    # be a mode nobody outside the test suite could turn on.
    from proxy_extract.cli import parse_backend_options

    options = parse_backend_options(["window=4", "process_res=728"])

    assert options == {"window": 4, "process_res": 728}
    assert get_backend("depth_anything_v3", **options).window == 4


def test_a_backend_option_that_is_not_a_pair_is_refused():
    from proxy_extract.cli import parse_backend_options

    with pytest.raises(SystemExit, match="KEY=VALUE"):
        parse_backend_options(["window"])


def test_a_checkpoint_name_survives_being_parsed():
    from proxy_extract.cli import parse_backend_options

    options = parse_backend_options([f"checkpoint={METRIC_CHECKPOINT}"])

    assert options == {"checkpoint": METRIC_CHECKPOINT}


def test_a_missing_package_explains_the_awkward_install(monkeypatch):
    monkeypatch.delitem(sys.modules, "depth_anything_3", raising=False)
    monkeypatch.delitem(sys.modules, "depth_anything_3.api", raising=False)

    real_import = __import__

    def refuse(name, *args, **kwargs):
        if name.startswith("depth_anything_3"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse)

    with pytest.raises(ImportError) as caught:
        DepthAnythingV3Backend(device="cpu").estimate(_frames(1))

    message = str(caught.value)
    assert "--no-deps" in message and "--ignore-requires-python" in message
    assert "einops" in message


def test_only_the_unreachable_submodules_get_stood_in_for(fake_da3, monkeypatch):
    model = fake_da3(_FakeModel())
    # pose_align is importable here; export is not. Only export should be faked.
    real = types.ModuleType("depth_anything_3.utils.pose_align")
    real.align_poses_umeyama = lambda *a, **k: "real"
    monkeypatch.setitem(sys.modules, "depth_anything_3.utils.pose_align", real)

    backend = DepthAnythingV3Backend(device="cpu")
    backend.estimate(_frames(1))

    assert backend._stubbed == ["depth_anything_3.utils.export"]
    assert sys.modules["depth_anything_3.utils.pose_align"] is real
    assert model.calls == [1]


def test_a_stub_that_is_actually_called_says_why_it_cannot_work(fake_da3):
    fake_da3(_FakeModel())
    backend = DepthAnythingV3Backend(device="cpu")
    backend.estimate(_frames(1))

    export = sys.modules["depth_anything_3.utils.export"]
    with pytest.raises(RuntimeError, match="optional dependencies"):
        export.export()
