"""Cutting delivered segments into SFT clips.

The arithmetic tests are the point of this file. A clip is wrong in exactly two
ways that no eye catches: it can run at the wrong speed, and it can overlap its
neighbour. Both look perfect frame by frame.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from proxy_extract import clips, delivery


def test_dropping_frames_keeps_the_speed_the_camera_saw():
    """30 fps into 24 must skip a frame every five, not renumber all of them."""
    windows = clips.plan_windows("seg_000000", 1800, source_fps=30.0, count=5)

    offsets = np.diff(np.asarray(windows[0].ordinals))
    assert set(np.unique(offsets)) == {1, 2}, "30 -> 24 keeps four of every five"
    assert offsets.mean() == pytest.approx(30.0 / 24.0, abs=0.01)

    # The wall-clock duration of the source frames a clip reads has to equal
    # the duration of the clip itself; that equality is what "same speed" is.
    span_seconds = (windows[0].stop - windows[0].start) / 30.0
    assert span_seconds == pytest.approx(clips.CLIP_FRAMES / 24.0, abs=0.05)


def test_a_matching_rate_takes_every_frame():
    window = clips.plan_windows("seg_000000", 1800, source_fps=24.0, count=5, fps=24.0)[0]
    assert np.all(np.diff(np.asarray(window.ordinals)) == 1)


def test_the_windows_are_spread_out_and_never_touch():
    windows = clips.plan_windows("seg_000000", 1800, source_fps=30.0, count=5)

    assert len(windows) == 5
    assert all(len(w.ordinals) == clips.CLIP_FRAMES for w in windows)

    gaps = [b.start - a.stop for a, b in zip(windows, windows[1:])]
    assert all(gap > 0 for gap in gaps), f"windows touch or overlap: {gaps}"
    assert max(gaps) - min(gaps) <= 1, f"unevenly spread: {gaps}"

    # Centred in its bucket, so the episode's first and last frames are not
    # systematically the only ones that never appear in any clip.
    assert windows[0].start > 0
    assert windows[-1].stop < 1800


def test_the_plan_is_the_same_every_time_it_is_asked_for():
    first = clips.plan_windows("seg_000012", 1800, source_fps=30.0)
    again = clips.plan_windows("seg_000012", 1800, source_fps=30.0)
    assert [w.ordinals for w in first] == [w.ordinals for w in again]


def test_an_episode_too_short_to_cut_says_so():
    with pytest.raises(clips.ClipError, match="spans"):
        clips.plan_windows("seg_000000", 400, source_fps=30.0, count=5)


def test_speeding_a_stream_up_is_refused_rather_than_invented():
    with pytest.raises(clips.ClipError, match="never recorded"):
        clips.plan_windows("seg_000000", 1800, source_fps=20.0, fps=24.0)


def test_a_clip_is_named_after_the_segment_it_came_from():
    assert clips.clip_name("seg_000123", 2) == "clip_000123_2"


# ------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def delivered(tmp_path_factory):
    """One short delivered segment, cut from a synthetic episode."""
    root = tmp_path_factory.mktemp("delivered")
    source = root / "video.mp4"
    _write_source(source, frames_count=200, fps=30.0)

    scene = root / "seg_000000"
    delivery.extract_scene(
        source,
        scene,
        config=delivery.DeliveryConfig(
            depth_backend="synthetic",
            semantic_backend="synthetic",
            size=(128, 72),
            chunk_frames=16,
            stabilise_block=8,
        ),
    )
    return scene


def _write_source(path, *, frames_count: int, fps: float) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 144)
    )
    rng = np.random.default_rng(0)
    for index in range(frames_count):
        frame = np.full((144, 256, 3), index % 251, dtype=np.uint8)
        frame[:40, :40] = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _cut_one(delivered, out, **kwargs):
    reports = clips.cut_scene(delivered, out, count=2, length=8, fps=24.0, **kwargs)
    return reports


def test_a_clip_holds_the_frames_its_report_claims(delivered, tmp_path):
    from proxy_extract.video import probe

    reports = _cut_one(delivered, tmp_path)
    assert len(reports) == 2

    clip = tmp_path / reports[0]["clip"]
    target = probe(clip / clips.TARGET_DIRNAME / clips.TARGET_NAME)
    duv = probe(clip / clips.PROXY_DIRNAME / clips.DUV_NAME)

    assert target.frames == duv.frames == 8
    assert (target.width, target.height) == (clips.TARGET_WIDTH, clips.TARGET_HEIGHT)
    assert (duv.width, duv.height) == (clips.DUV_WIDTH, clips.DUV_HEIGHT)
    assert round(target.fps) == 24

    written = json.loads((clip / clips.CLIP_REPORT_NAME).read_text())
    assert written["frames"] == len(written["source_ordinals"]) == 8
    assert written["scene"] == "seg_000000"


def test_the_anchor_is_the_targets_own_first_frame(delivered, tmp_path):
    """Not a re-render of it, and not the source frame at another size."""
    import cv2

    from proxy_extract.video import read_frames

    reports = _cut_one(delivered, tmp_path)
    clip = tmp_path / reports[0]["clip"]

    anchor = cv2.imread(str(clip / clips.TARGET_DIRNAME / clips.ANCHOR_NAME))[:, :, ::-1]
    first = read_frames(clip / clips.TARGET_DIRNAME / clips.TARGET_NAME, limit=1)[0]

    assert anchor.shape == first.shape == (clips.TARGET_HEIGHT, clips.TARGET_WIDTH, 3)
    # The anchor is lossless PNG and the target is x264, so they differ by the
    # codec and nothing else: same picture, not the same bytes.
    assert np.abs(anchor.astype(int) - first.astype(int)).mean() < 12


def test_the_duv_only_ever_shows_codes_that_were_predicted(delivered, tmp_path):
    """A resize would blend the class palette into colours nothing means."""
    from proxy_extract import proxy
    from proxy_extract.video import read_frames

    reports = _cut_one(delivered, tmp_path)
    clip = tmp_path / reports[0]["clip"]
    duv = np.stack(read_frames(clip / clips.PROXY_DIRNAME / clips.DUV_NAME))

    palette = {(g, b) for g, b in proxy._PROXY_GB_STANDARD11.values()}
    palette.add(proxy._PROXY_EGO_GB)
    seen = {tuple(pair) for pair in np.unique(duv[:, :, :, 1:].reshape(-1, 2), axis=0)}
    assert seen <= palette, f"invented colours: {sorted(seen - palette)}"


def test_one_duv_pixel_is_exactly_one_4x4_block_of_the_target():
    """The reduction has to be the block one, not the nearest-neighbour fallback.

    1280x720 is not a multiple of 336x192, and `contract` silently samples one
    pixel in fifteen when the factor is not integral. Here a near object covers
    a fifth of a block; nearest-neighbour returns whatever it lands on, and the
    median must return the depth that most of the block actually is.
    """
    from proxy_extract import contract

    metres = np.full((720, 1280), 40.0, dtype=np.float32)
    metres[:, ::5] = 0.5  # a picket fence, a fifth of every block

    reduced = clips._to_duv_grid(metres, contract.downsample_depth)

    assert reduced.shape == (clips.DUV_HEIGHT, clips.DUV_WIDTH)
    assert np.median(reduced) == pytest.approx(40.0), "the near minority won the vote"


def test_a_minority_class_does_not_take_the_block():
    from proxy_extract import contract

    ids = np.zeros((720, 1280), dtype=np.uint8)
    ids[:, ::5] = 7  # thin vegetation over sky, never more than a fifth

    reduced = clips._to_duv_grid(ids, contract.downsample_semantic)

    assert reduced.shape == (clips.DUV_HEIGHT, clips.DUV_WIDTH)
    assert set(np.unique(reduced)) == {0}


def test_a_second_pass_leaves_a_finished_clip_alone(delivered, tmp_path):
    reports = _cut_one(delivered, tmp_path)
    clip = tmp_path / reports[0]["clip"]
    target = clip / clips.TARGET_DIRNAME / clips.TARGET_NAME
    before = target.stat().st_mtime_ns

    again = _cut_one(delivered, tmp_path, resume=True)

    assert target.stat().st_mtime_ns == before
    assert [item["clip"] for item in again] == [item["clip"] for item in reports]


def test_a_half_written_clip_is_cut_again(delivered, tmp_path):
    reports = _cut_one(delivered, tmp_path)
    clip = tmp_path / reports[0]["clip"]
    (clip / clips.TARGET_DIRNAME / clips.ANCHOR_NAME).unlink()

    assert not clips.already_cut(clip, 8)
    _cut_one(delivered, tmp_path, resume=True)
    assert clips.already_cut(clip, 8)


def test_the_audit_counts_what_is_whole(delivered, tmp_path):
    _cut_one(delivered, tmp_path)
    summary = clips.audit_clips(tmp_path, 8)
    assert summary["complete"] == 2
    assert summary["incomplete"] == 0


def test_a_segment_that_was_never_delivered_is_refused(tmp_path):
    (tmp_path / "seg_000000").mkdir()
    with pytest.raises(clips.ClipError, match="extraction_report"):
        clips.cut_scene(tmp_path / "seg_000000", tmp_path / "clips")


def test_the_command_cuts_only_the_segments_that_are_finished(delivered, tmp_path):
    """The CLI takes its segment list from the manifest, not from a glob.

    A segment still being written has frames that are about to change, and a
    clip cut from one is indistinguishable afterwards from a clip cut from a
    finished one.
    """
    from proxy_extract import cli

    out = tmp_path / "out"
    out.mkdir()
    (out / "seg_000000").symlink_to(delivered)
    (out / "seg_000001").mkdir()  # started, nothing in it yet

    assignments = delivery.assign_scenes(
        [("ep0", delivered.parent / "video.mp4", None), ("ep1", delivered.parent / "video.mp4", None)]
    )
    delivery.write_manifest(out, assignments)

    code = cli.main(
        [
            "clips", "--out", str(out), "--clips-out", str(tmp_path / "clips"),
            "--per-scene", "2", "--frames", "8", "--fps", "24", "--quiet",
        ]
    )

    assert code == 0
    cut = sorted(path.name for path in (tmp_path / "clips").glob("clip_*"))
    assert cut == ["clip_000000_0", "clip_000000_1"], "an unfinished segment was cut"


def test_the_actions_are_cut_to_the_frames_the_clip_kept():
    window = clips.Window(scene="seg_000000", index=0, ordinals=(0, 2, 4))
    payload = [{"frame": index} for index in range(10)]
    assert clips._slice_actions(payload, window.ordinals) == [
        {"frame": 0}, {"frame": 2}, {"frame": 4}
    ]


def test_a_wrapped_action_list_keeps_its_wrapper():
    window = clips.Window(scene="seg_000000", index=0, ordinals=(1, 3))
    payload = {"fps": 30, "actions": [{"frame": index} for index in range(10)]}
    out = clips._slice_actions(payload, window.ordinals)
    assert out["fps"] == 30
    assert out["actions"] == [{"frame": 1}, {"frame": 3}]


def test_something_shorter_than_the_episode_is_not_mistaken_for_per_frame():
    """Slicing a 3-entry list by frame index would pair actions with the wrong pictures."""
    window = clips.Window(scene="seg_000000", index=0, ordinals=(100, 101))
    payload = {"keys": ["w", "a", "s"], "actions": [{"i": n} for n in range(200)]}
    out = clips._slice_actions(payload, window.ordinals)
    assert out["keys"] == ["w", "a", "s"]
    assert out["actions"] == [{"i": 100}, {"i": 101}]
