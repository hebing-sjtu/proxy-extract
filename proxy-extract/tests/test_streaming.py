"""The streaming rewrite has to produce what the batch code produced.

Every stage that used to take an episode-sized stack now takes frames one at a
time, and each one of those changes is only safe if it is *equivalent*, not
merely similar. A stabiliser that differs slightly at its window seams, or a
resume that restarts with truncated temporal context, would still write four
videos of the right length in the right format, and nothing downstream could
tell. So the equivalences are asserted directly, against the batch functions
they replaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import delivery, frames, streaming
from proxy_extract.depth.scale import apply_range_guard
from proxy_extract.depth.synthetic import SyntheticDepthBackend
from proxy_extract.taxonomy import NUM_CLASSES
from proxy_extract.temporal import flicker_rate, flow_compensated_pair

from .test_delivery import SIZE, write_clip


def synthetic_stack(frames_count: int, height: int = 24, width: int = 32):
    """Depth, labels and a flow guide with enough structure to move flow around."""
    rng = np.random.default_rng(20260902)
    depth = (rng.random((frames_count, height, width), dtype=np.float32) * 40.0) + 0.5
    labels = rng.integers(0, NUM_CLASSES, (frames_count, height, width), dtype=np.uint8)
    guide = [
        rng.integers(0, 255, (height, width), dtype=np.uint8) for _ in range(frames_count)
    ]
    return depth, labels, guide


def drain(window, depth, labels, guide):
    out = []
    for index in range(len(depth)):
        out.extend(window.push(depth[index], labels[index], guide[index] if guide else None))
    out.extend(window.close())
    return out


@pytest.mark.parametrize("block", [1, 3, 7, 64])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_the_sliding_window_reproduces_the_batch_pass_exactly(block, radius):
    """A frame is only released once it has the neighbours the batch call gave it.

    Bit-exact rather than close: the window computes the same optical flow
    between the same pair of guide frames, so any difference here would mean a
    frame was released early, with part of its window missing.
    """
    depth, labels, guide = synthetic_stack(20)
    want_depth, want_labels = flow_compensated_pair(
        depth, labels, guide_frames=guide, radius=radius
    )

    got = drain(
        streaming.WindowStabiliser(radius=radius, block=block, flow_compensate=True),
        depth,
        labels,
        guide,
    )

    assert [ordinal for ordinal, _, _ in got] == list(range(len(depth)))
    np.testing.assert_array_equal(np.stack([value for _, _, value in got]), want_labels)
    np.testing.assert_array_equal(np.stack([value for _, value, _ in got]), want_depth)


def test_the_window_also_matches_with_flow_compensation_off():
    depth, labels, _ = synthetic_stack(12)
    want_depth, want_labels = flow_compensated_pair(depth, labels, guide_frames=None, radius=2)

    got = drain(
        streaming.WindowStabiliser(radius=2, block=5, flow_compensate=False), depth, labels, None
    )

    np.testing.assert_array_equal(np.stack([value for _, _, value in got]), want_labels)
    np.testing.assert_array_equal(np.stack([value for _, value, _ in got]), want_depth)


def test_the_window_holds_only_the_frames_its_radius_needs():
    """The point of the rewrite: memory bounded by the window, not the episode."""
    depth, labels, guide = synthetic_stack(200)
    window = streaming.WindowStabiliser(radius=2, block=8, flow_compensate=True)

    widest = 0
    for index in range(len(depth)):
        list(window.push(depth[index], labels[index], guide[index]))
        widest = max(widest, len(window._depth))

    assert widest <= 8 + 2 * 2 + 1


def test_a_frame_is_refused_without_the_guide_it_was_promised():
    depth, labels, _ = synthetic_stack(3)
    window = streaming.WindowStabiliser(radius=1, flow_compensate=True)
    with pytest.raises(ValueError, match="guide"):
        list(window.push(depth[0], labels[0], None))


def test_the_streamed_range_guard_reports_what_the_batch_one_did():
    depth, _, _ = synthetic_stack(9)
    near, far = delivery.DELIVERY_NEAR_METRES, delivery.DELIVERY_FAR_METRES

    want_clipped, want_stats = apply_range_guard(depth, near=near, far=far)
    guard = streaming.RangeGuard(near=near, far=far)
    got_clipped = np.stack([guard.apply(frame) for frame in depth])
    got_stats = guard.stats()

    np.testing.assert_array_equal(got_clipped, want_clipped)
    assert got_stats["clipped_near_fraction"] == want_stats["clipped_near_fraction"]
    assert got_stats["clipped_far_fraction"] == want_stats["clipped_far_fraction"]
    # The median comes from a histogram of float16 bit patterns, which is exact
    # for the depth this pipeline stores and within a step of the float32 the
    # batch version measured.
    assert got_stats["median_metres"] == pytest.approx(want_stats["median_metres"], rel=1e-3)


def test_the_range_guard_survives_the_restart_it_will_meet():
    depth, _, _ = synthetic_stack(11)
    near, far = delivery.DELIVERY_NEAR_METRES, delivery.DELIVERY_FAR_METRES

    whole = streaming.RangeGuard(near=near, far=far)
    for frame in depth:
        whole.apply(frame)

    first = streaming.RangeGuard(near=near, far=far)
    for frame in depth[:6]:
        first.apply(frame)
    second = streaming.RangeGuard(near=near, far=far)
    second.restore(first.state())
    for frame in depth[6:]:
        second.apply(frame)

    assert second.stats() == whole.stats()


def test_the_flicker_meter_matches_the_rate_it_replaces():
    _, labels, _ = synthetic_stack(15)
    meter = streaming.FlickerMeter()
    for frame in labels:
        meter.push(frame)
    assert meter.rate == flicker_rate(labels)


# ------------------------------------------------------------------- the store


def test_frames_round_trip_through_the_store(tmp_path):
    frames.make_dirs(tmp_path)
    depth = np.array([[0.0, 1.5], [800.0, 42.25]], dtype=np.float32)
    labels = np.array([[0, 3], [10, 7]], dtype=np.uint8)
    rgb = np.dstack([np.full((2, 2), 7), np.full((2, 2), 8), np.full((2, 2), 9)]).astype(np.uint8)

    frames.write_array(tmp_path, "depth", 0, depth)
    frames.write_array(tmp_path, "semantic", 0, labels)
    frames.write_image(tmp_path, "color", 0, rgb)

    np.testing.assert_allclose(frames.read_array(tmp_path, "depth", 0), depth, rtol=1e-3)
    np.testing.assert_array_equal(frames.read_array(tmp_path, "semantic", 0), labels)
    np.testing.assert_array_equal(frames.read_image(tmp_path, "color", 0), rgb)


def test_depth_is_stored_finely_enough_to_beat_the_video_it_becomes(tmp_path):
    """float16 has to cost less than the 8-bit quantiser downstream of it."""
    frames.make_dirs(tmp_path)
    metres = np.geomspace(0.1, 8000.0, 512, dtype=np.float32).reshape(16, 32)

    frames.write_array(tmp_path, "depth", 0, metres)
    read_back = frames.read_array(tmp_path, "depth", 0)

    relative = np.abs(read_back - metres) / metres
    assert relative.max() < 1e-3


def test_a_half_written_frame_is_not_counted_as_present(tmp_path):
    """The resume point is a file count, so a truncated write must not be one."""
    frames.make_dirs(tmp_path)
    for ordinal in range(4):
        frames.write_array(tmp_path, "depth", ordinal, np.zeros((2, 2), dtype=np.float32))
    partial = frames.frame_path(tmp_path, "depth", 4)
    partial.with_name(partial.name + frames.PARTIAL_SUFFIX).write_bytes(b"truncated")

    assert frames.contiguous_count(tmp_path, "depth") == 4


def test_counting_stops_at_the_first_hole(tmp_path):
    frames.make_dirs(tmp_path)
    for ordinal in (0, 1, 2, 5, 6):
        frames.write_array(tmp_path, "depth", ordinal, np.zeros((2, 2), dtype=np.float32))

    assert frames.contiguous_count(tmp_path, "depth") == 3

    frames.discard_from(tmp_path, "depth", 3)
    assert frames.contiguous_count(tmp_path, "depth") == 3
    assert not frames.frame_path(tmp_path, "depth", 5).exists()


# -------------------------------------------------------------------- resuming


class StopsPartWay:
    """A depth backend that goes away mid-episode, the way a killed worker does."""

    name = "synthetic"

    def __init__(self, after: int) -> None:
        self._inner = SyntheticDepthBackend()
        self._after = after
        self._seen = 0

    def estimate(self, batch, *, cameras=None):
        if self._seen >= self._after:
            raise RuntimeError("worker went away")
        self._seen += len(batch)
        return self._inner.estimate(batch, cameras=cameras)


def delivery_config(**overrides):
    return delivery.DeliveryConfig(
        depth_backend="synthetic",
        semantic_backend="synthetic",
        size=SIZE,
        chunk_frames=5,
        stabilise_block=4,
        **overrides,
    )


@pytest.fixture
def long_episode(tmp_path):
    path = tmp_path / "ep" / "video.mp4"
    path.parent.mkdir(parents=True)
    write_clip(path, 30)
    return path


def scene_frames(scene_dir):
    count = frames.contiguous_count(scene_dir, "depth")
    return {
        "depth": [frames.read_array(scene_dir, "depth", i) for i in range(count)],
        "semantic": [frames.read_array(scene_dir, "semantic", i) for i in range(count)],
        "color": [frames.read_image(scene_dir, "color", i) for i in range(count)],
        "duv": [frames.read_image(scene_dir, "duv", i) for i in range(count)],
    }


def test_a_resumed_episode_is_identical_to_an_uninterrupted_one(long_episode, tmp_path):
    """The seam has to be invisible, not small.

    Resume re-runs the models over `temporal_radius` frames before the point it
    stopped at, precisely so the first unwritten frame gets the same left-hand
    neighbours an uninterrupted run would have given it. Without that the frames
    either side of a restart would be stabilised against a truncated window -
    a defect that no frame count or format check would ever surface.
    """
    config = delivery_config(flow_compensate=True, flow_downscale=1)

    whole = tmp_path / "whole"
    reference = delivery.extract_scene(long_episode, whole, config=config)

    broken = tmp_path / "broken"
    with pytest.raises(RuntimeError, match="worker went away"):
        delivery.extract_scene(
            long_episode, broken, config=config, depth_backend=StopsPartWay(after=12)
        )
    stopped_at = frames.contiguous_count(broken, "depth")
    assert 0 < stopped_at < reference["frames"]

    resumed = delivery.extract_scene(long_episode, broken, config=config)

    assert resumed["frames"] == reference["frames"]
    want, got = scene_frames(whole), scene_frames(broken)
    for stream in frames.STREAMS:
        for ordinal, (expected, actual) in enumerate(zip(want[stream], got[stream])):
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"{stream} frame {ordinal} differs after resume"
            )


def test_resuming_discards_frames_written_under_different_settings(long_episode, tmp_path):
    """Half an episode at one setting and half at another is undetectable later."""
    scene = tmp_path / "seg_000000"
    delivery.extract_scene(long_episode, scene, config=delivery_config(temporal_radius=2))

    report = delivery.extract_scene(
        long_episode, scene, config=delivery_config(temporal_radius=3)
    )

    assert report["config"]["temporal_radius"] == 3
    assert frames.contiguous_count(scene, "depth") == report["frames"]


def test_the_working_directory_does_not_outlive_the_scene(long_episode, tmp_path):
    scene = tmp_path / "seg_000000"
    delivery.extract_scene(long_episode, scene, config=delivery_config())

    assert not frames.stage_dir_for(scene).exists()
    for stream in frames.STREAMS:
        assert frames.contiguous_count(scene, stream) == 30


def test_dropping_the_frames_leaves_the_videos(long_episode, tmp_path):
    scene = tmp_path / "seg_000000"
    report = delivery.extract_scene(long_episode, scene, config=delivery_config(keep_frames=()))

    assert report["frames_kept"] == []
    assert not frames.frames_dir_for(scene).exists()
    assert delivery.already_done(scene)


def test_keeping_one_stream_drops_the_other_three(long_episode, tmp_path):
    """The expensive choice is per stream: only depth outlives its own video."""
    scene = tmp_path / "seg_000000"
    report = delivery.extract_scene(
        long_episode, scene, config=delivery_config(keep_frames=("depth",))
    )

    assert report["frames_kept"] == ["depth"]
    assert frames.contiguous_count(scene, "depth") == report["frames"]
    for dropped in ("color", "semantic", "duv"):
        assert not frames.stream_dir(scene, dropped).exists()
    assert delivery.already_done(scene)


def test_an_unknown_stream_is_refused_rather_than_ignored():
    from proxy_extract.cli import parse_kept_streams

    assert parse_kept_streams("depth,semantic") == ("depth", "semantic")
    assert parse_kept_streams("none") == ()
    with pytest.raises(SystemExit, match="dpeth"):
        parse_kept_streams("dpeth")
