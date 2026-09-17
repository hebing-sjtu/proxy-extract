"""The SAM 2 consistency layer, exercised against a stand-in for the tracker.

Nothing here downloads a checkpoint. What is worth testing is not SAM 2 - it is
the join, which is where the reasoning lives: that a masklet gets **one** class
for the whole clip, that the class is the trunk's pooled opinion rather than any
single frame's, and that pixels no masklet reached are left to the trunk instead
of being claimed by the nearest one.

The stand-in propagates each seed mask unchanged, which is the useful
idealisation: a perfect tracker. Under it, any label flicker that survives is
this module's doing rather than the model's.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest

from proxy_extract.semantic import get_refiner
from proxy_extract.semantic.base import SemanticResult
from proxy_extract.semantic.sam2 import NO_OWNER, Sam2ConsistencyRefiner
from proxy_extract.taxonomy import BUILDING_STRUCTURE, HUMAN, ROAD_PAVED, VEHICLE

FRAMES = 12
H, W = 48, 64


@pytest.fixture(autouse=True)
def _torch(monkeypatch):
    """Real torch where it exists, a no-op stand-in where it does not.

    The refiner only asks torch for two context managers and a dtype, none of
    which affect the label arithmetic under test, so a machine without torch
    should still run these. Preferring the real one where it is installed keeps
    the test honest on the deployment target.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        fake = types.ModuleType("torch")
        fake.bfloat16 = "bfloat16"
        fake.inference_mode = contextlib.nullcontext
        fake.autocast = lambda **_kwargs: contextlib.nullcontext()
        fake.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", fake)


class _FakePredictor:
    """A perfect tracker: every seed mask propagates unchanged.

    Records the prompts it was given so the tests can assert on how many
    masklets were started and on which frames.
    """

    def __init__(self) -> None:
        self.prompts: list[tuple[int, int, np.ndarray]] = []
        self.propagations: list[tuple[int, int]] = []

    def init_state(self, video_path, **_kwargs):
        import os

        return {"frames": len(os.listdir(video_path))}

    def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
        self.prompts.append((frame_idx, obj_id, np.asarray(mask, dtype=bool)))

    def propagate_in_video(
        self, *, inference_state, start_frame_idx, max_frame_num_to_track
    ):
        self.propagations.append((start_frame_idx, max_frame_num_to_track))
        total = inference_state["frames"]
        # The real predictor's own arithmetic, which is inclusive at both ends:
        #   end = min(start + max_frame_num_to_track, num_frames - 1)
        #   for frame_idx in range(start, end + 1)
        # Getting this off by one here would have hidden the fact that the
        # re-seeding pass depends on the start frame already being tracked.
        stop = min(start_frame_idx + max_frame_num_to_track, total - 1)
        live = [(obj, mask) for born, obj, mask in self.prompts if born <= start_frame_idx]
        for index in range(start_frame_idx, stop + 1):
            ids = [obj for obj, _mask in live]
            # Logits, not booleans: the refiner thresholds at zero, the way the
            # real predictor's output requires.
            logits = np.stack(
                [np.where(mask, 4.0, -4.0) for _obj, mask in live]
            ).astype(np.float32)
            yield index, ids, logits


@pytest.fixture
def fake_sam2(monkeypatch):
    def install(predictor):
        module = types.ModuleType("sam2")
        video = types.ModuleType("sam2.sam2_video_predictor")

        class SAM2VideoPredictor:
            @staticmethod
            def from_pretrained(_checkpoint, **_kwargs):
                return predictor

        video.SAM2VideoPredictor = SAM2VideoPredictor
        monkeypatch.setitem(sys.modules, "sam2", module)
        monkeypatch.setitem(sys.modules, "sam2.sam2_video_predictor", video)
        return predictor

    return install


def _frames(count=FRAMES):
    # Not uniform grey: the staging step JPEG-encodes these, and an encoder
    # given a constant image is a poor exercise of the path.
    base = np.tile(np.arange(W, dtype=np.uint8), (H, 1))
    return [np.dstack([base, base + index, base]) for index in range(count)]


def _flickering_labels(*, flip_between=(ROAD_PAVED, BUILDING_STRUCTURE)):
    """A big region whose class alternates every frame - the worst case.

    README's `test_a_majority_vote_alone_cannot_remove_it` shows an odd-length
    window vote cannot fix this, which is exactly why this module exists.
    """
    labels = np.zeros((FRAMES, H, W), dtype=np.uint8)
    for index in range(FRAMES):
        labels[index, :, :] = flip_between[index % 2]
    return labels


def _base(labels):
    return SemanticResult(labels=labels, meta={"backend": "panoptic"})


# --------------------------------------------------- the thing it exists for


def test_perfect_alternation_cannot_survive_a_masklet(fake_sam2):
    """One label per masklet makes frame-to-frame flicker unrepresentable.

    Not suppressed - unrepresentable. The masklet has a single class for the
    clip, so there is no per-frame value left for it to alternate between.
    """
    predictor = fake_sam2(_FakePredictor())
    refiner = Sam2ConsistencyRefiner(device="cpu", reseed_every=FRAMES)

    result = refiner.refine(_frames(), _base(_flickering_labels()))

    changes = (result.labels[1:] != result.labels[:-1]).mean()
    assert changes == 0.0, "the labels still change between frames"
    assert result.meta["sam2_covered_fraction"] == pytest.approx(1.0)
    assert predictor.prompts, "nothing was seeded, so nothing was tested"


def test_the_class_is_the_pooled_vote_not_the_first_frames(fake_sam2):
    """A masklet born on a frame the trunk got wrong still ends up right.

    Seeding from frame 0 and keeping frame 0's label would be the obvious
    implementation and would propagate that frame's mistake across the clip.
    """
    fake_sam2(_FakePredictor())
    labels = np.full((FRAMES, H, W), VEHICLE, dtype=np.uint8)
    labels[0] = BUILDING_STRUCTURE  # the seed frame is the one that is wrong

    result = Sam2ConsistencyRefiner(device="cpu", reseed_every=FRAMES).refine(
        _frames(), _base(labels)
    )

    assert np.all(result.labels == VEHICLE)
    assert result.meta["sam2_masklet_classes"] == {int(VEHICLE): 1}


def test_a_thing_keeps_its_own_masklet_inside_a_region(fake_sam2):
    """Components are found per class, so a car against a wall is two masklets.

    Over the whole frame they would be one component of "not background" and
    would vote as whichever owns more pixels, losing the smaller one for the
    rest of the clip.
    """
    fake_sam2(_FakePredictor())
    labels = np.full((FRAMES, H, W), BUILDING_STRUCTURE, dtype=np.uint8)
    labels[:, 10:30, 10:30] = VEHICLE

    result = Sam2ConsistencyRefiner(device="cpu", reseed_every=FRAMES).refine(
        _frames(), _base(labels)
    )

    assert result.meta["sam2_masklets"] == 2
    assert np.all(result.labels[:, 15, 15] == VEHICLE)
    assert np.all(result.labels[:, 40, 50] == BUILDING_STRUCTURE)


def test_the_smaller_masklet_keeps_a_contested_pixel(fake_sam2):
    """Painted largest first, so a thing inside a region survives it.

    Area order rather than taxonomy priority because a masklet's class is not
    known while the votes are still being counted.
    """
    fake_sam2(_FakePredictor())
    labels = np.full((FRAMES, H, W), ROAD_PAVED, dtype=np.uint8)
    labels[:, 20:24, 20:24] = HUMAN

    result = Sam2ConsistencyRefiner(
        device="cpu", reseed_every=FRAMES, min_area_fraction=1e-4
    ).refine(_frames(), _base(labels))

    assert np.all(result.labels[:, 21, 21] == HUMAN)


# ------------------------------------------------------------- what it leaves


def test_pixels_no_masklet_reached_keep_the_trunks_labels(fake_sam2):
    """Uncovered is uncovered: `temporal.py` still has work to do afterwards.

    Claiming them for the nearest masklet would be worse than flicker - it
    would be confident and wrong over a region nothing looked at.
    """
    fake_sam2(_FakePredictor())
    labels = np.full((FRAMES, H, W), ROAD_PAVED, dtype=np.uint8)
    # A speck far below the seeding floor, so no masklet is ever started for it.
    labels[:, 0, 0] = VEHICLE

    refiner = Sam2ConsistencyRefiner(
        device="cpu", reseed_every=FRAMES, min_area_fraction=0.05
    )
    result = refiner.refine(_frames(), _base(labels))

    assert np.all(result.labels[:, 0, 0] == VEHICLE), "the trunk's answer was discarded"
    assert result.meta["sam2_covered_fraction"] < 1.0


def test_a_frame_with_nothing_big_enough_to_seed_is_survivable(fake_sam2):
    """No prompts means no propagation: the real predictor raises on an empty
    prompt set inside its preflight."""
    predictor = fake_sam2(_FakePredictor())
    labels = np.zeros((FRAMES, H, W), dtype=np.uint8)

    refiner = Sam2ConsistencyRefiner(
        device="cpu", reseed_every=FRAMES, min_area_fraction=1.5
    )
    result = refiner.refine(_frames(), _base(labels))

    assert predictor.propagations == []
    assert result.meta["sam2_masklets"] == 0
    assert np.array_equal(result.labels, labels)


# ---------------------------------------------------------------- re-seeding


def test_content_that_enters_later_gets_its_own_masklet(fake_sam2):
    """A car driving into frame at second three has to be tracked at all."""
    predictor = fake_sam2(_FakePredictor())
    labels = np.full((FRAMES, H, W), ROAD_PAVED, dtype=np.uint8)
    labels[6:, 10:30, 10:30] = VEHICLE

    result = Sam2ConsistencyRefiner(device="cpu", reseed_every=3).refine(
        _frames(), _base(labels)
    )

    assert result.meta["sam2_seed_frames"] != [0], "nothing was picked up after frame 0"
    assert VEHICLE in result.meta["sam2_masklet_classes"]
    assert [start for start, _ in predictor.propagations] == [0, 3, 6, 9]


def test_a_region_already_covered_is_not_seeded_again(fake_sam2):
    """Otherwise the object cap fills with copies of the road.

    The coverage threshold is below 1.0 because a masklet's boundary drifts
    from the trunk's, and exact coverage would never be reached.
    """
    fake_sam2(_FakePredictor())
    labels = np.full((FRAMES, H, W), ROAD_PAVED, dtype=np.uint8)

    result = Sam2ConsistencyRefiner(device="cpu", reseed_every=2).refine(
        _frames(), _base(labels)
    )

    assert result.meta["sam2_masklets"] == 1
    assert result.meta["sam2_seed_frames"] == [0]


def test_the_object_cap_is_respected_and_reported(fake_sam2):
    """The cap decides whether a clip fits on the card, so hitting it is a
    fact the report has to carry rather than something to infer."""
    fake_sam2(_FakePredictor())
    labels = np.zeros((FRAMES, H, W), dtype=np.uint8)
    # Eight separate blocks of distinct classes, so eight components exist.
    for index in range(8):
        labels[:, :, index * 8 : index * 8 + 6] = index + 1

    result = Sam2ConsistencyRefiner(
        device="cpu", reseed_every=FRAMES, max_objects=3, min_area_fraction=1e-4
    ).refine(_frames(), _base(labels))

    assert result.meta["sam2_masklets"] == 3
    assert result.meta["sam2_masklets_at_cap"] is True


# -------------------------------------------------------------- housekeeping


def test_the_batch_length_is_recorded(fake_sam2):
    """SAM 2's memory spans one call, so a batched run has seams and the report
    is the only place that says how long the tracked span actually was."""
    fake_sam2(_FakePredictor())

    result = Sam2ConsistencyRefiner(device="cpu", reseed_every=FRAMES).refine(
        _frames(), _base(_flickering_labels())
    )

    assert result.meta["sam2_frames_in_call"] == FRAMES
    assert result.meta["backend"] == "panoptic", "the trunk's meta must survive"


def test_the_staging_directory_does_not_outlive_the_call(fake_sam2, tmp_path):
    import tempfile

    predictor = fake_sam2(_FakePredictor())
    before = set(_scratch_dirs(tempfile.gettempdir()))

    Sam2ConsistencyRefiner(device="cpu", reseed_every=FRAMES).refine(
        _frames(), _base(_flickering_labels())
    )

    assert predictor.prompts
    assert set(_scratch_dirs(tempfile.gettempdir())) == before


def _scratch_dirs(where):
    from pathlib import Path

    return [p.name for p in Path(where).glob("sam2-*")]


def test_the_refiner_is_reachable_by_name():
    refiner = get_refiner("sam2", reseed_every=8)

    assert isinstance(refiner, Sam2ConsistencyRefiner)
    assert refiner.reseed_every == 8


def test_no_owner_is_negative():
    """It indexes a class array, so a sentinel of 0 would silently mean the
    first class rather than "nobody"."""
    assert NO_OWNER < 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reseed_every": 0}, "reseed_every"),
        ({"max_objects": 0}, "max_objects"),
        ({"coverage_threshold": 0.0}, "coverage_threshold"),
        ({"coverage_threshold": 1.5}, "coverage_threshold"),
    ],
)
def test_nonsense_settings_are_refused_before_anything_loads(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Sam2ConsistencyRefiner(**kwargs)


def test_a_base_of_the_wrong_length_is_refused(fake_sam2):
    fake_sam2(_FakePredictor())
    refiner = Sam2ConsistencyRefiner(device="cpu")

    with pytest.raises(ValueError, match="frames"):
        refiner.refine(_frames(4), _base(_flickering_labels()))


def test_labels_of_the_wrong_shape_are_refused(fake_sam2):
    fake_sam2(_FakePredictor())
    refiner = Sam2ConsistencyRefiner(device="cpu")
    wrong = np.zeros((FRAMES, H // 2, W), dtype=np.uint8)

    with pytest.raises(ValueError, match="labels"):
        refiner.refine(_frames(), _base(wrong))


def test_a_missing_package_says_how_to_get_it(monkeypatch):
    monkeypatch.delitem(sys.modules, "sam2", raising=False)
    monkeypatch.delitem(sys.modules, "sam2.sam2_video_predictor", raising=False)
    real_import = __import__

    def refuse(name, *args, **kwargs):
        if name.startswith("sam2"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse)

    with pytest.raises(ImportError, match="github.com/facebookresearch/sam2"):
        Sam2ConsistencyRefiner(device="cpu")._load()
