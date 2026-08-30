"""Video decoding, kept deliberately thin.

Everything downstream works on uint8 RGB `HxWx3` arrays. OpenCV is the only
decoder here because it is the one dependency already present on both the Mac
workstation and the H800 image; swapping in decord or PyAV later only needs to
preserve the two functions below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int


def probe(path: Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            path=Path(path),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def read_frames(
    path: Path,
    *,
    size: tuple[int, int] | None = None,
    limit: int | None = None,
    grayscale: bool = False,
) -> list[np.ndarray]:
    """Decode sequentially to a list of RGB (or gray) uint8 arrays.

    Sequential decode rather than seeking: these clips are short, and seeking
    on H.264 without an exact-frame index silently lands on the nearest
    keyframe, which would desynchronise the high/low comparison.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while limit is None or len(frames) < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            if size is not None and (bgr.shape[1], bgr.shape[0]) != size:
                shrinking = size[0] * size[1] < bgr.shape[0] * bgr.shape[1]
                bgr = cv2.resize(
                    bgr, size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
                )
            if grayscale:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            else:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"decoded zero frames from {path}")
    return frames
