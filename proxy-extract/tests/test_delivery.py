"""The 720p delivery scene has to be readable exactly as DATA_F.md says.

Three properties carry the format and are checked by decoding what was written
rather than by inspecting what was intended:

- the four videos agree on frame count, because their whole value is being
  frame-aligned with each other;
- semantic IDs survive the encode bit-exactly, because a YUV round trip
  silently invents classes and nothing downstream can detect it;
- depth round-trips to metres within the quantiser's own step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proxy_extract import delivery, proxy
from proxy_extract.taxonomy import NUM_STANDARD11
from proxy_extract.video import probe

SIZE = (192, 128)  # a small stand-in for 1280x720; the code is size-agnostic


def write_clip(path, frames: int, width: int = 288, height: int = 192) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    rng = np.random.default_rng(20260901)
    for index in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[: height // 2, :, 2] = 200  # sky-ish top half
        frame[height // 2 :, :, 1] = 150
        frame[60:140, 40 + index % 40 : 120 + index % 40] = 210
        frame += rng.integers(0, 10, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "ep" / "video.mp4"
    path.parent.mkdir(parents=True)
    write_clip(path, 24)
    (path.parent / "annotations.tar").write_bytes(b"not-a-real-tar-but-copied-verbatim")
    return path


@pytest.fixture
def scene(episode, tmp_path):
    out = tmp_path / "seg_000000"
    report = delivery.extract_scene(
        episode,
        out,
        config=delivery.DeliveryConfig(
            depth_backend="synthetic",
            semantic_backend="synthetic",
            size=SIZE,
            chunk_frames=7,
            flow_compensate=False,
        ),
    )
    return out, report


def decode_rgb(path, *, limit=None):
    """Decode losslessly as RGB, which is the only correct way to read these."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while limit is None or len(frames) < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(bgr[:, :, ::-1].copy())
    finally:
        capture.release()
    return np.stack(frames)


def test_the_four_videos_live_under_proxy_and_the_source_tar_beside_it(scene):
    """What we derived goes in `proxy/`; what the corpus gave us stays out of it."""
    out, _ = scene

    for name in ("color.mp4", "depth.mp4", "semantic.mp4", "duv.mp4"):
        assert (out / "proxy" / name).is_file(), name
        assert (out / "proxy" / name).stat().st_size > 0, name
        assert not (out / name).exists(), f"{name} must not also sit in the scene root"
    assert (out / "annotations.tar").read_bytes() == b"not-a-real-tar-but-copied-verbatim"
    assert (out / "extraction_report.json").is_file()


def test_every_frame_is_delivered_at_the_requested_size(scene):
    out, report = scene

    assert report["frames"] == 24, "no windowing: the whole episode is delivered"
    assert report["size"] == list(SIZE)
    for name in ("color.mp4", "depth.mp4", "semantic.mp4", "duv.mp4"):
        info = probe(out / "proxy" / name)
        assert info.frames == 24, name
        assert (info.width, info.height) == SIZE, name


def test_source_frame_rate_is_preserved(scene):
    _, report = scene

    assert report["fps"] == pytest.approx(30.0), "the source rate, not DATA_F.md's 24"


def test_semantic_ids_survive_the_encode_bit_exactly(scene):
    out, report = scene
    decoded = decode_rgb(out / "proxy" / "semantic.mp4")

    assert decoded.shape == (24, SIZE[1], SIZE[0], 3)
    assert not decoded[:, :, :, 0].any(), "R must be 0; a non-zero R means a YUV decode"
    assert not decoded[:, :, :, 1].any(), "G must be 0"
    ids = decoded[:, :, :, 2]
    assert int(ids.max()) < NUM_STANDARD11

    fractions = report["semantic"]["class_fractions"]
    assert fractions, "the report has to say which classes are present"


def test_depth_round_trips_within_the_quantiser_step(scene):
    out, _ = scene
    import cv2

    capture = cv2.VideoCapture(str(out / "proxy" / "depth.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok

    grey = frame[:, :, 0]
    metres = proxy.decode_depth_frame(grey)
    valid = metres > 0
    assert valid.any(), "the synthetic backend fills the frame, so depth must decode"

    # One code is a fixed ratio over a log scale, so the tolerance is relative.
    step = (proxy.DEPTH_VIDEO_FAR_METRES / proxy.DEPTH_VIDEO_NEAR_METRES) ** (1 / 255)
    reencoded = proxy.encode_depth_frame(metres)
    assert np.array_equal(reencoded[valid], grey[valid])
    assert step < 1.05


def test_depth_codes_survive_the_gray_encode_bit_exactly(tmp_path):
    """Every one of the 256 codes has to come back unchanged.

    ffmpeg does not hand `gray` to x264 as-is; it converts to `yuvj420p` and
    carries the values in the luma plane. That is only lossless because `yuvj`
    is *full range*, so 0..255 maps one-to-one. Plain `yuv420p` would compress
    them into 16..235 and quantise, silently corrupting depth everywhere - so
    this pins the behaviour rather than trusting the conversion.
    """
    import cv2

    height, width = 128, 192
    frames = []
    for shift in range(6):
        frame = np.zeros((height, width), np.uint8)
        for code in range(256):
            row0, row1 = (code * height) // 256, ((code + 1) * height) // 256
            frame[row0:row1, :] = (code + shift) % 256
        frames.append(frame)

    path = tmp_path / "depth.mp4"
    encoder = proxy.open_encoder(path, width, height, 30.0, kind="depth")
    for frame in frames:
        encoder.write(frame)
    encoder.close()

    capture = cv2.VideoCapture(str(path))
    decoded = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
            assert np.array_equal(blue, green) and np.array_equal(green, red)
            decoded.append(blue)
    finally:
        capture.release()

    assert len(decoded) == len(frames)
    for index, (want, got) in enumerate(zip(frames, decoded)):
        assert np.array_equal(want, got), f"depth codes changed in frame {index}"


def test_semantic_ids_round_trip_through_the_real_encoder(tmp_path):
    """Adjacent IDs are the case a YUV pipeline breaks, so encode them adjacent."""
    import cv2

    height, width = 96, 160
    ids = np.zeros((height, width), np.uint8)
    for cls in range(NUM_STANDARD11):
        ids[:, cls * width // NUM_STANDARD11 : (cls + 1) * width // NUM_STANDARD11] = cls
    # Single-pixel stripes: the hardest thing for chroma subsampling to keep.
    ids[::2, ::2] = (ids[::2, ::2] + 1) % NUM_STANDARD11

    path = tmp_path / "semantic.mp4"
    encoder = proxy.open_encoder(path, width, height, 30.0, kind="semantic")
    for _ in range(4):
        encoder.write(proxy.encode_semantic_frame(ids))
    encoder.close()

    capture = cv2.VideoCapture(str(path))
    try:
        ok, bgr = capture.read()
    finally:
        capture.release()
    assert ok
    assert np.array_equal(bgr[:, :, 0], ids), "ids must return unchanged in the B channel"
    assert not bgr[:, :, 1].any() and not bgr[:, :, 2].any()


def test_proxy_marks_sky_with_the_reserved_red_code(scene):
    out, _ = scene
    proxy_frames = decode_rgb(out / "proxy" / "duv.mp4", limit=1)
    semantic_frames = decode_rgb(out / "proxy" / "semantic.mp4", limit=1)

    red = proxy_frames[0, :, :, 0]
    ids = semantic_frames[0, :, :, 2]
    sky = ids == 0
    if sky.any():
        assert (red[sky] == proxy.PROXY_SKY_CODE).all()
    assert (red[~sky] <= proxy.PROXY_MAX_CODE).all()


def test_report_records_the_encoding_the_reader_needs(scene):
    _, report = scene

    assert report["depth"]["encoding"] == "h264-logz-gray8"
    assert report["depth"]["near_metres"] == 0.1
    assert report["depth"]["far_metres"] == 256.0
    assert report["duv_depth_inverted"] is False, "duv R runs forward; depth.mp4 does not"
    assert report["semantic"]["taxonomy"] == "cwm12"


def test_up_to_scale_depth_is_refused_rather_than_shipped(episode, tmp_path):
    class UpToScale:
        name = "up_to_scale"

        def estimate(self, frames, *, cameras=None):
            from proxy_extract.depth.base import DepthResult

            stacked = np.ones((len(frames), *frames[0].shape[:2]), dtype=np.float32)
            return DepthResult(depth=stacked, metric=False, meta={})

    with pytest.raises(delivery.DeliveryError, match="up-to-scale"):
        delivery.extract_scene(
            episode,
            tmp_path / "refused",
            config=delivery.DeliveryConfig(
                semantic_backend="synthetic", size=SIZE, chunk_frames=8, flow_compensate=False
            ),
            depth_backend=UpToScale(),
        )


# --------------------------------------------------------- scene numbering


def test_scenes_are_numbered_by_sample_id_not_discovery_order():
    from pathlib import Path

    episodes = [
        ("zzz", Path("/data/zzz/video.mp4"), None),
        ("aaa", Path("/data/aaa/video.mp4"), None),
        ("mmm", Path("/data/mmm/video.mp4"), None),
    ]
    assigned = delivery.assign_scenes(episodes)

    assert [item.scene for item in assigned] == ["seg_000000", "seg_000001", "seg_000002"]
    assert [item.sample_id for item in assigned] == ["aaa", "mmm", "zzz"]


def test_numbering_is_stable_under_sharding():
    from pathlib import Path

    from proxy_extract.pipeline import shard

    episodes = [(f"s{i:03d}", Path(f"/data/s{i:03d}/video.mp4"), None) for i in range(20)]
    assigned = delivery.assign_scenes(episodes)

    recombined = sorted(
        (item.index, item.sample_id)
        for worker in range(4)
        for item in shard(assigned, worker, 4)
    )
    assert recombined == [(item.index, item.sample_id) for item in assigned]


def test_directory_naming_leaves_room_for_the_whole_corpus():
    """Six digits, because ABot ships 30,969 episodes and four would wrap."""
    from pathlib import Path

    assert delivery.scene_dir_for(Path("/out"), 0).name == "seg_000000"
    assert delivery.scene_dir_for(Path("/out"), 1999).name == "seg_001999"
    assert delivery.scene_dir_for(Path("/out"), 30968).name == "seg_030968"


def test_duplicate_sample_ids_are_refused():
    from pathlib import Path

    with pytest.raises(delivery.DeliveryError, match="repeat"):
        delivery.assign_scenes(
            [("same", Path("/a/video.mp4"), None), ("same", Path("/b/video.mp4"), None)]
        )


def test_episodes_from_videos_keys_abot_on_the_parent_directory(tmp_path):
    first = tmp_path / "sample_a" / "video.mp4"
    second = tmp_path / "sample_b" / "video.mp4"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.touch()
    (first.parent / "annotations.tar").touch()

    episodes = delivery.episodes_from_videos([first, second])

    assert [sample for sample, _, _ in episodes] == ["sample_a", "sample_b"]
    assert episodes[0][2] == first.parent / "annotations.tar"
    assert episodes[1][2] is None, "a missing tar is recorded as absent, not invented"


def test_manifest_records_provenance_for_every_scene(tmp_path):
    from pathlib import Path

    assigned = delivery.assign_scenes(
        [("bbb", Path("/data/bbb/video.mp4"), Path("/data/bbb/annotations.tar"))]
    )
    path = delivery.write_manifest(tmp_path, assigned)
    payload = json.loads(path.read_text())

    assert payload["count"] == 1
    assert payload["scenes"][0]["scene"] == "seg_000000"
    assert payload["scenes"][0]["sample_id"] == "bbb"
    assert payload["scenes"][0]["source_video"] == "/data/bbb/video.mp4"


def test_resume_rejects_a_scene_whose_videos_are_short(scene):
    out, report = scene

    assert delivery.already_done(out)

    report["frames"] = report["frames"] + 5
    (out / "extraction_report.json").write_text(json.dumps(report))
    assert not delivery.already_done(out), "a truncated scene must be redone, not accepted"


def test_resume_survives_a_video_it_cannot_even_open(scene):
    """A worker killed mid-encode leaves an mp4 with no `moov` atom.

    `--resume` asks this question about every scene, so answering with an
    exception would lose a whole shard over one damaged directory.
    """
    out, _ = scene
    payload = (out / "proxy" / "duv.mp4").read_bytes()
    (out / "proxy" / "duv.mp4").write_bytes(payload[: len(payload) // 3])

    assert delivery.already_done(out) is False


def test_audit_counts_missing_and_damaged_without_raising(scene, tmp_path):
    from pathlib import Path

    out, _ = scene
    root = out.parent
    delivery.write_manifest(
        root,
        delivery.assign_scenes(
            [
                ("aaa", Path(out / "video.mp4"), None),
                ("bbb", Path("/gone/video.mp4"), None),
            ]
        ),
    )
    payload = (out / "proxy" / "color.mp4").read_bytes()
    (out / "proxy" / "color.mp4").write_bytes(payload[:512])

    summary = delivery.audit(root)

    assert summary["expected"] == 2
    assert summary["complete"] == 0
    assert summary["incomplete"] == 1, "the damaged scene is incomplete, not missing"
    assert summary["missing"] == 1
    assert summary["incomplete_scenes"] == ["seg_000000"]


def test_a_placeholder_run_says_so_in_its_own_report(scene):
    """The failure this prevents already happened once.

    Synthetic output is structurally identical to real output — same codecs,
    same class ids, same report shape — so a contact sheet made from it was
    read as evidence that the real segmenter was broken. The report now carries
    the fact rather than leaving it to be inferred from a backend name.
    """
    _, report = scene
    assert report["deliverable"] is False
    assert sorted(report["placeholder_backends"]) == ["depth=synthetic", "semantic=synthetic"]


def test_a_placeholder_run_warns_while_it_writes(episode, tmp_path):
    with pytest.warns(delivery.PlaceholderOutput, match="must not be delivered"):
        delivery.extract_scene(
            episode,
            tmp_path / "seg_000000",
            config=delivery.DeliveryConfig(
                depth_backend="synthetic",
                semantic_backend="synthetic",
                size=SIZE,
                chunk_frames=7,
                flow_compensate=False,
            ),
        )


def test_a_real_backend_pair_is_marked_deliverable(episode, tmp_path, monkeypatch):
    """Only the fabricating backends trip the flag, not every non-default one."""

    class NamedOnly:
        def __init__(self, name):
            self.name = name

    config = delivery.DeliveryConfig(depth_backend="mapanything", semantic_backend="standard11")
    assert delivery.placeholder_backends(NamedOnly("mapanything"), NamedOnly("standard11"), config) == []
    assert delivery.placeholder_backends(NamedOnly("mapanything"), NamedOnly("synthetic"), config) == [
        "semantic=synthetic"
    ]


def test_a_delivered_scene_renders_to_a_contact_sheet(scene, tmp_path):
    """The delivery format is unviewable by construction; this is the way in.

    `semantic.mp4` holds class ids 0-10 in the blue channel, so it plays as
    near-black and tells a reviewer nothing. The sheet has to come back with
    the palette applied, not the raw ids.
    """
    from proxy_extract.preview import render_scene_preview

    out, _ = scene
    sheet = render_scene_preview(out, tmp_path / "sheet.png", frames=3, width=160)

    import cv2

    image = cv2.imread(str(sheet))
    assert image is not None and image.size > 0
    # Three panels wide, and the semantic third must be coloured rather than
    # the near-black the raw video decodes to.
    third = image.shape[1] // 3
    semantic_panel = image[:, third : 2 * third]
    assert semantic_panel.max() > 100


def test_the_scene_preview_also_writes_a_video(scene, tmp_path):
    from proxy_extract.preview import render_scene_preview

    out, _ = scene
    path = render_scene_preview(out, tmp_path / "sheet.mp4", width=160)
    assert path.stat().st_size > 0


def test_the_scene_preview_refuses_a_directory_that_is_not_a_scene(tmp_path):
    from proxy_extract.preview import render_scene_preview

    with pytest.raises(FileNotFoundError, match="no proxy/color.mp4"):
        render_scene_preview(tmp_path, tmp_path / "sheet.png")


def _assignment(episode, index=0):
    return delivery.SceneAssignment(
        index=index, sample_id=f"sample{index}", video=episode, annotations=None
    )


def test_a_missing_dependency_stops_the_run_even_with_keep_going(episode, tmp_path, monkeypatch):
    """The failure this prevents cost 1800 episodes of wall time.

    `--keep-going` exists so one unreadable episode costs one scene. A backend
    whose package is not installed is not that: every remaining episode fails
    identically, so skipping burns the corpus and reports a run that produced
    nothing.
    """
    def explode(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'mapanything'")

    monkeypatch.setattr(delivery, "extract_scene", explode)

    assignments = [_assignment(episode, i) for i in range(5)]
    with pytest.raises(delivery.DeliveryError, match="environment problem"):
        delivery.deliver_dataset(
            assignments,
            tmp_path,
            config=delivery.DeliveryConfig(
                depth_backend="synthetic", semantic_backend="synthetic", size=SIZE
            ),
            on_error="skip",
        )


def test_the_message_says_it_is_not_the_episodes_fault(episode, tmp_path, monkeypatch):
    monkeypatch.setattr(
        delivery,
        "extract_scene",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'mapanything'")),
    )
    with pytest.raises(delivery.DeliveryError) as caught:
        delivery.deliver_dataset(
            [_assignment(episode)],
            tmp_path,
            config=delivery.DeliveryConfig(
                depth_backend="synthetic", semantic_backend="synthetic", size=SIZE
            ),
            on_error="skip",
        )
    message = str(caught.value)
    assert "mapanything" in message
    assert "--keep-going deliberately does not cover it" in message


def test_an_ordinary_bad_episode_is_still_skipped(episode, tmp_path, monkeypatch):
    """The environment rule must not swallow what keep-going is actually for."""
    calls = []

    def flaky(video, scene_dir, **kwargs):
        calls.append(video)
        if len(calls) == 1:
            raise ValueError("this episode has no moov atom")
        return {"scene": scene_dir.name}

    monkeypatch.setattr(delivery, "extract_scene", flaky)

    reports = delivery.deliver_dataset(
        [_assignment(episode, 0), _assignment(episode, 1)],
        tmp_path,
        config=delivery.DeliveryConfig(
            depth_backend="synthetic", semantic_backend="synthetic", size=SIZE
        ),
        on_error="skip",
    )
    assert "failed" in reports[0] and "moov" in reports[0]["failed"]
    assert reports[1]["scene"] == delivery.scene_dir_for(tmp_path, 1).name
