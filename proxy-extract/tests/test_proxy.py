"""The delivery encodings from DATA_F.md.

The semantic round trip is the test that matters. If it ever starts failing,
the encode has fallen back to a YUV pipeline and every ID map in the dataset is
quietly wrong at class boundaries.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from proxy_extract import contract, proxy
from proxy_extract import taxonomy as tax

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def read_video(path, *, grayscale: bool):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if grayscale
                else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            )
    finally:
        capture.release()
    return frames


def build_condition_root(root, frames: int, taxonomy: str, *, driving: bool = False):
    """A small hand-made condition_root, so the encoders are tested alone."""
    root.mkdir(parents=True, exist_ok=True)
    height, width = contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH
    top = {"standard11": tax.NUM_STANDARD11, "coarse6": tax.NUM_COARSE6}.get(
        taxonomy, tax.NUM_CLASSES
    )

    for ordinal in range(frames):
        depth = np.full((height, width), 5.0, dtype=np.float32)
        depth[:, : width // 3] = 0.5
        depth[:, width // 3 : 2 * width // 3] = 40.0
        depth[:20, :] = 0.0  # sky band: no depth

        ids = np.zeros((height, width), dtype=np.uint8)
        for cls in range(top):
            ids[:, cls * (width // top) : (cls + 1) * (width // top)] = cls

        contract.write_frame(root, ordinal, depth, ids)

    (root / "extraction_report.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "semantic": {"taxonomy": taxonomy, "hero_split": {"driving": driving}},
            }
        )
    )


def test_depth_round_trips_through_the_log_encoding():
    metres = np.array([[0.1, 0.5, 1.0, 10.0, 100.0, 256.0]], dtype=np.float32)

    grey = proxy.encode_depth_frame(metres)
    back = proxy.decode_depth_frame(grey)

    # 8-bit over a 0.1-256 m log range is coarse, so compare in relative terms.
    assert grey[0, 0] == 255, "the near plane must be the brightest code"
    relative = np.abs(back[0, :5] - metres[0, :5]) / metres[0, :5]
    assert np.all(relative < 0.02), f"relative error too high: {relative}"


def test_depth_marks_invalid_and_far_as_zero():
    metres = np.array([[0.0, 256.0, 1e6]], dtype=np.float32)

    grey = proxy.encode_depth_frame(metres)

    assert list(grey[0]) == [0, 0, 0]
    assert list(proxy.decode_depth_frame(grey)[0]) == [0.0, 0.0, 0.0]


def test_semantic_ids_survive_the_lossless_encode(tmp_path):
    root = tmp_path / "cond"
    build_condition_root(root, 6, "standard11")

    summary = proxy.write_videos(root, tmp_path / "out", kinds=("semantic",), fps=24)

    decoded = read_video(summary["videos"]["semantic"], grayscale=False)
    assert len(decoded) == 6
    _, want_ids = contract.read_frame(root, 0)
    for frame in decoded:
        assert np.array_equal(frame[:, :, 2], want_ids), "IDs changed in the round trip"
        assert not frame[:, :, 0].any(), "R channel must stay zero"
        assert not frame[:, :, 1].any(), "G channel must stay zero"


def test_proxy_marks_sky_with_the_reserved_code():
    depth = np.array([[0.0, 1.0]], dtype=np.float32)
    ids = np.array([[tax.S11_SKY, tax.S11_ROAD]], dtype=np.uint8)

    frame = proxy.compose_proxy_frame(depth, ids)

    assert frame[0, 0, 0] == proxy.PROXY_SKY_CODE
    assert frame[0, 1, 0] != proxy.PROXY_SKY_CODE, "road must not collide with the sky sentinel"
    assert tuple(frame[0, 0, 1:]) == (255, 255)
    assert tuple(frame[0, 1, 1:]) == (255, 255)


def test_proxy_uses_ego_colour_only_when_driving():
    depth = np.ones((1, 1), dtype=np.float32)
    ids = np.array([[tax.S11_VEHICLE]], dtype=np.uint8)

    parked = proxy.compose_proxy_frame(depth, ids, driving=False)
    ego = proxy.compose_proxy_frame(depth, ids, driving=True)

    assert tuple(parked[0, 0, 1:]) == (64, 0)
    assert tuple(ego[0, 0, 1:]) == (128, 0)


def test_proxy_depth_direction_is_selectable():
    depth = np.array([[1.0, 1000.0]], dtype=np.float32)
    ids = np.zeros((1, 2), dtype=np.uint8) + tax.S11_ROAD

    inverted = proxy.compose_proxy_frame(depth, ids, inverted_depth=True)
    forward = proxy.compose_proxy_frame(depth, ids, inverted_depth=False)

    assert inverted[0, 0, 0] > inverted[0, 1, 0], "inverted: near is the higher code"
    assert forward[0, 0, 0] < forward[0, 1, 0], "forward: near is the lower code"
    assert int(inverted[0, 0, 0]) + int(forward[0, 0, 0]) == proxy.PROXY_MAX_CODE


def test_proxy_depth_runs_forward_by_default():
    """DATA_F.md fixes the direction through where it puts the sky sentinel.

    Sky is 255, so valid depth has to ascend towards the far plane for the
    sentinel to sit just beyond it. Inverting would put the nearest surface at
    254 immediately below sky at 255 — the collision the sentinel exists to
    prevent. The depth video runs the other way for the same reason: its
    sentinel is 0, so its codes descend.
    """
    depth = np.array([[proxy.PROXY_NEAR_METRES, proxy.PROXY_FAR_METRES]], dtype=np.float32)
    ids = np.zeros((1, 2), dtype=np.uint8) + tax.S11_ROAD

    frame = proxy.compose_proxy_frame(depth, ids)

    assert frame[0, 0, 0] == 0, "the near plane is code 0"
    assert frame[0, 1, 0] == proxy.PROXY_MAX_CODE, "the far plane is code 254"
    assert proxy.PROXY_SKY_CODE == proxy.PROXY_MAX_CODE + 1, "sky sits directly past far"


def test_the_two_videos_place_their_sentinels_at_the_same_end():
    """Opposite directions, same principle: the sentinel is just past far."""
    near = np.array([[proxy.DEPTH_VIDEO_NEAR_METRES]], dtype=np.float32)
    far = np.array([[proxy.DEPTH_VIDEO_FAR_METRES]], dtype=np.float32)

    # depth.mp4: descends to the far plane, sentinel 0 below it.
    assert proxy.encode_depth_frame(near)[0, 0] == 255
    assert proxy.encode_depth_frame(far)[0, 0] == 0

    # duv.mp4: ascends to the far plane, sentinel 255 above it.
    ids = np.zeros((1, 1), dtype=np.uint8) + tax.S11_ROAD
    at_near = proxy.compose_proxy_frame(
        np.array([[proxy.PROXY_NEAR_METRES]], dtype=np.float32), ids
    )
    at_far = proxy.compose_proxy_frame(
        np.array([[proxy.PROXY_FAR_METRES]], dtype=np.float32), ids
    )
    assert at_near[0, 0, 0] < at_far[0, 0, 0] < proxy.PROXY_SKY_CODE


def test_coarse6_projects_onto_the_delivery_ids():
    ids = np.array([[tax.C6_HERO, tax.C6_NPC, tax.C6_ROAD, tax.C6_VEGETATION]], dtype=np.uint8)

    projected = proxy.to_standard11(ids, "coarse6")

    assert list(projected[0]) == [tax.S11_PLAYER, tax.S11_PED, tax.S11_ROAD, tax.S11_VEGETATION]


def test_write_videos_emits_all_three_and_reports(tmp_path):
    root = tmp_path / "cond"
    build_condition_root(root, 5, "coarse6", driving=True)

    summary = proxy.write_videos(root, tmp_path / "out", fps=30)

    assert set(summary["videos"]) == {"depth", "semantic", "proxy"}
    assert summary["frames"] == 5
    assert summary["fps"] == 30
    assert summary["driving"] is True
    for kind, path in summary["videos"].items():
        decoded = read_video(path, grayscale=(kind == "depth"))
        assert len(decoded) == 5, f"{kind} has {len(decoded)} frames"


def test_write_videos_needs_a_report(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()

    with pytest.raises(proxy.EncodeError, match="extraction_report"):
        proxy.write_videos(root)
