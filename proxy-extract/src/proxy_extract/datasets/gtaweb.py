"""Reader for the gta-web clip corpus (DATA_F.md, contract v3).

This corpus carries engine ground truth for both depth and semantics, which
makes it the scoring set rather than a processing target. The decoders below
exist because both GT streams are stored in ways that decode *plausibly* but
wrongly under the obvious settings, and neither failure raises:

  - depth is `h264-logz-gray8`. Read through a colour path it still yields an
    image, just one whose values have been through a YUV round trip. One grey
    level is 3.13% of depth, so a range expansion nobody notices moves every
    metre reading.
  - semantics are `libx264rgb` / `rgb24` at crf 0 with `(R, G, B) = (0, 0, id)`.
    Decoded as YUV 4:2:0 the chroma subsampling averages neighbouring ids, and
    ids 5 and 6 — road and ground, which share long borders — differ by one.

So every decode here is checked against a property the correct decode must
have, and raises rather than returning something usable-looking. `probe_decode`
runs those checks on a single frame if you want the diagnosis without paying
for the whole clip.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..taxonomy import NUM_STANDARD11

# DATA_F.md "Depth（米制）". Note near=0.1 is closer than the DUV contract's
# 0.3: a seventh of the code space describes distances CWM cannot represent.
DEPTH_NEAR_METRES = 0.1
DEPTH_FAR_METRES = 256.0
_LOG_FAR = math.log(DEPTH_FAR_METRES)
_LOG_SPAN = _LOG_FAR - math.log(DEPTH_NEAR_METRES)

FRAMES_PER_CLIP = 124
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FPS = 24

# gray == 0 means sky or no hit. The DUV encoder has no invalid code — 256 m
# encodes to 0 there — so sky has to be written as the far plane. Writing 0.0
# metres instead would send it through the log encoding as the *nearest*
# representable depth, i.e. sky in your face.
SKY_METRES = DEPTH_FAR_METRES


class DecodeError(ValueError):
    """A stream decoded to something the format guarantees it cannot be."""


@dataclass(frozen=True)
class Clip:
    """One entry of a directory's `clips.json`."""

    tag: str
    frame_start: int
    frame_end: int
    color: str | None = None
    depth: str | None = None
    semantic: str | None = None
    proxy: str | None = None
    subject: str | None = None
    turn: bool = False
    raw: dict = None  # type: ignore[assignment]

    @property
    def frames(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def is_driving(self) -> bool:
        return self.tag.startswith("drive")


def _first_key(payload: dict, *names: str, default=None):
    for name in names:
        if name in payload:
            return payload[name]
    return default


def load_clips(directory: Path) -> list[Clip]:
    """Parse `clips.json`, tolerating the two file-name spellings in the set."""
    directory = Path(directory)
    payload = json.loads((directory / "clips.json").read_text())
    entries = payload["clips"] if isinstance(payload, dict) else payload

    clips = []
    for entry in entries:
        clips.append(
            Clip(
                tag=str(entry.get("tag", "")),
                frame_start=int(_first_key(entry, "frameStart", "frame_start", default=0)),
                frame_end=int(_first_key(entry, "frameEnd", "frame_end", default=0)),
                color=_first_key(entry, "color", "colorFile"),
                depth=_first_key(entry, "depth", "depthFile"),
                semantic=_first_key(entry, "semantic", "semanticFile"),
                proxy=_first_key(entry, "proxy", "proxyFile"),
                subject=entry.get("subject"),
                turn=bool(entry.get("turn", False)),
                raw=entry,
            )
        )
    return clips


# ------------------------------------------------------------------- depth


def gray_to_metres(gray: np.ndarray, *, sky: float = SKY_METRES) -> np.ndarray:
    """Invert DATA_F.md's log-z encoding. `gray == 0` becomes `sky`."""
    gray = np.asarray(gray)
    metres = np.exp(_LOG_FAR - (gray.astype(np.float64) / 255.0) * _LOG_SPAN)
    return np.where(gray > 0, metres, sky).astype(np.float32)


def metres_to_gray(metres: np.ndarray) -> np.ndarray:
    """Forward direction, for round-trip tests only."""
    metres = np.clip(np.asarray(metres, dtype=np.float64), DEPTH_NEAR_METRES, DEPTH_FAR_METRES)
    q16 = np.round((_LOG_FAR - np.log(metres)) / _LOG_SPAN * 65535.0)
    gray = np.maximum(1.0, np.round(q16 * 255.0 / 65535.0))
    return np.where(q16 == 0, 0, gray).astype(np.uint8)


def quantisation_step() -> float:
    """Relative depth error of one grey level: ~0.0313, i.e. 3.13%.

    The precision ceiling of the whole corpus. The DUV encoding is ~299x finer
    over the same span, so nothing downstream is the limiting factor.
    """
    return math.expm1(_LOG_SPAN / 255.0)


def decode_depth(path: Path, *, limit: int | None = None) -> np.ndarray:
    """Decode a `h264-logz-gray8` depth clip to metric depth, `(frames, H, W)`.

    Raises if the three colour channels disagree. For a genuinely grey stream
    they cannot: the decoder replicates luma into all three. Disagreement means
    the frames went through a chroma path, and the values are no longer the
    engine's grey codes.
    """
    frames = []
    for bgr in _iter_bgr(path, limit=limit):
        if not (np.array_equal(bgr[:, :, 0], bgr[:, :, 1]) and np.array_equal(bgr[:, :, 1], bgr[:, :, 2])):
            raise DecodeError(
                f"depth channels differ in {path}: decoded through a chroma path, "
                "so the grey codes are no longer the engine's. Force a gray8 decode."
            )
        frames.append(gray_to_metres(bgr[:, :, 0]))
    if not frames:
        raise DecodeError(f"decoded zero frames from {path}")
    return np.stack(frames)


# ---------------------------------------------------------------- semantic


def decode_semantic(path: Path, *, limit: int | None = None) -> np.ndarray:
    """Decode a lossless-RGB semantic clip to class ids, `(frames, H, W)`.

    Checks both properties the format guarantees — `R == G == 0` and every id
    inside the 11-class range. A YUV round trip breaks the first almost
    everywhere and the second wherever two classes meet, so this catches the
    wrong decode on frame one rather than after a thousand clips.
    """
    frames = []
    for bgr in _iter_bgr(path, limit=limit):
        blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        stray = int(np.count_nonzero(green) + np.count_nonzero(red))
        if stray:
            raise DecodeError(
                f"semantic frame in {path} has {stray} pixels with non-zero R or G; the "
                "format is (0, 0, id), so this decode was not lossless RGB. Force rgb24."
            )
        top = int(blue.max(initial=0))
        if top >= NUM_STANDARD11:
            raise DecodeError(
                f"semantic id {top} in {path} exceeds the {NUM_STANDARD11 - 1} the standard "
                "defines; ids have been interpolated, which means a resampling or YUV decode."
            )
        frames.append(blue.astype(np.uint8))
    if not frames:
        raise DecodeError(f"decoded zero frames from {path}")
    return np.stack(frames)


def probe_decode(depth_path: Path | None = None, semantic_path: Path | None = None) -> dict:
    """Run the decode checks on one frame each and report, without raising.

    For answering "is this corpus readable the way the docs claim" before
    committing a batch run to it.
    """
    report: dict = {}
    if depth_path is not None:
        try:
            depth = decode_depth(depth_path, limit=1)[0]
            finite = depth[depth < SKY_METRES]
            report["depth"] = {
                "ok": True,
                "shape": list(depth.shape),
                "sky_fraction": round(float(np.mean(depth >= SKY_METRES)), 4),
                "min_metres": round(float(finite.min()), 4) if finite.size else None,
                "max_metres": round(float(finite.max()), 4) if finite.size else None,
                "distinct_levels": int(np.unique(depth).size),
            }
        except (DecodeError, FileNotFoundError) as error:
            report["depth"] = {"ok": False, "error": str(error)}

    if semantic_path is not None:
        try:
            semantic = decode_semantic(semantic_path, limit=1)[0]
            report["semantic"] = {
                "ok": True,
                "shape": list(semantic.shape),
                "classes_present": sorted(int(c) for c in np.unique(semantic)),
            }
        except (DecodeError, FileNotFoundError) as error:
            report["semantic"] = {"ok": False, "error": str(error)}

    return report


def lossy_depth_suspicion(depth_metres: np.ndarray) -> dict:
    """Evidence about whether the depth mp4 was compressed lossily.

    DATA_F.md states crf 0 for the semantic stream and says nothing about
    depth. If depth went through ordinary lossy H.264, ringing at object
    borders shifts grey levels, and one level is 3.13% of depth.

    A lossless grey stream can only hold 256 distinct values. Substantially
    more than that means the decoder is producing intermediate values, which a
    correct pipeline never would.
    """
    distinct = int(np.unique(depth_metres).size)
    return {
        "distinct_values": distinct,
        "representable_max": 256,
        "lossless": distinct <= 256,
        "relative_step": round(quantisation_step(), 5),
    }


# ------------------------------------------------------------------ track


@dataclass(frozen=True)
class Track:
    """`track_<stamp>.json`: per-frame camera and player pose for a session."""

    camera: list[dict]
    player: list[dict]

    def slice(self, clip: Clip) -> "Track":
        return Track(
            camera=self.camera[clip.frame_start : clip.frame_end],
            player=self.player[clip.frame_start : clip.frame_end],
        )


def load_track(path: Path) -> Track:
    payload = json.loads(Path(path).read_text())
    return Track(
        camera=list(payload.get("camera", {}).get("frames", [])),
        player=list(payload.get("player", {}).get("frames", [])),
    )


# ----------------------------------------------------------------- helpers


def _iter_bgr(path: Path, *, limit: int | None = None):
    """Decode frames at native resolution, never resampling.

    Deliberately not `video.read_frames`: that resizes with INTER_AREA when
    shrinking, which averages pixels. On an id map averaging invents classes
    that do not exist, and on log-encoded depth it averages in the wrong space.
    Resampling belongs in `contract`, which knows to vote for ids and take
    medians for depth.
    """
    import cv2

    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        count = 0
        while limit is None or count < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            count += 1
            yield bgr
    finally:
        capture.release()
