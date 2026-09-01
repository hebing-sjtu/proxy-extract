"""Batched inference has to be a memory optimisation and nothing else.

The reduction happens before calibration and before the temporal passes, which
is only safe because the block reducers commute with a positive scale. If that
ever stops holding, these tests fail rather than quietly shifting depth in long
episodes.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import contract
from proxy_extract.cameras import CameraTrack
from proxy_extract.pipeline import ExtractionConfig, extract_clip
from proxy_extract.video import iter_frames, read_frames


def write_clip(path, frames: int, width: int = 320, height: int = 192) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
    rng = np.random.default_rng(20260901)
    for index in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :, 0] = (index * 3) % 256
        frame[40:120, 40 + index % 50 : 120 + index % 50] = 200
        frame += rng.integers(0, 12, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def clip(tmp_path):
    # 230 decoded frames reduces to 214 = 124 + 90, so truncation to a whole
    # number of windows is exercised alongside the batching.
    path = tmp_path / "clip.mp4"
    write_clip(path, 230)
    return path


def test_iter_frames_matches_read_frames(clip):
    whole = read_frames(clip, size=(160, 96))
    batched = [f for batch in iter_frames(clip, size=(160, 96), chunk=17) for f in batch]

    assert len(batched) == len(whole)
    assert all(np.array_equal(a, b) for a, b in zip(whole, batched))


def test_iter_frames_respects_limit_and_chunk(clip):
    batches = list(iter_frames(clip, size=(160, 96), limit=40, chunk=15))

    assert [len(b) for b in batches] == [15, 15, 10]


def test_iter_frames_rejects_zero_chunk(clip):
    with pytest.raises(ValueError, match="chunk must be"):
        list(iter_frames(clip, chunk=0))


def test_chunked_extraction_is_byte_identical(clip, tmp_path):
    common = dict(depth_backend="synthetic", semantic_backend="synthetic")
    whole = extract_clip(clip, tmp_path / "whole", config=ExtractionConfig(**common))
    chunked = extract_clip(
        clip, tmp_path / "chunked", config=ExtractionConfig(**common, chunk_frames=37)
    )

    assert whole["frames"] == chunked["frames"] == 214
    assert whole["inference_batches"] == 1
    assert chunked["inference_batches"] == 7

    for ordinal in range(whole["frames"]):
        want_depth, want_ids = contract.read_frame(tmp_path / "whole", ordinal)
        got_depth, got_ids = contract.read_frame(tmp_path / "chunked", ordinal)
        assert np.array_equal(want_depth, got_depth), f"depth differs at frame {ordinal}"
        assert np.array_equal(want_ids, got_ids), f"labels differ at frame {ordinal}"

    assert whole["semantic"]["flicker_after"] == chunked["semantic"]["flicker_after"]


def test_chunk_smaller_than_window_still_truncates_to_windows(clip, tmp_path):
    report = extract_clip(
        clip,
        tmp_path / "tiny",
        config=ExtractionConfig(
            depth_backend="synthetic", semantic_backend="synthetic", chunk_frames=8
        ),
    )

    assert report["frames"] == 214
    assert report["decoded_frames"] == 230


def test_chunking_refuses_a_gt_camera_track(clip, tmp_path):
    track = CameraTrack(
        cam2world=np.tile(np.eye(4), (230, 1, 1)),
        intrinsics=np.tile(np.eye(3), (230, 1, 1)),
        metric=True,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        extract_clip(
            clip,
            tmp_path / "cameras",
            config=ExtractionConfig(
                depth_backend="synthetic", semantic_backend="synthetic", chunk_frames=37
            ),
            cameras=track,
        )
