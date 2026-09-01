"""Encode a condition_root into the three delivery videos DATA_F.md defines.

The condition_root is what code-world-model consumes: raw float32 depth and an
8-bit ID map per frame. The videos here are the gta-web delivery format, and
their encodings are fixed by that document rather than chosen:

    depth_*.mp4      inverted log-z in 8-bit grey, near 0.1 m, far 256 m
    semantic_*.mp4   lossless RGB, (R, G, B) = (0, 0, id)
    proxy_*.mp4      R = forward log-z over near 0.1 / far 8000, sky 255,
                     G/B = semantic colour

Two of those are load-bearing details rather than preferences. The semantic
video must be encoded losslessly in RGB: pushing small integer IDs through a
YUV pipeline lets chroma subsampling blend neighbouring IDs into classes that
were never predicted, and nothing downstream can detect it. And the depth
channel is inverted, so bright means near.

ffmpeg does the writing because OpenCV's VideoWriter cannot ask for
`libx264rgb`, which is the only reason the IDs survive the round trip.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contract
from . import taxonomy as tax

# DATA_F.md's depth video range. Note this is *not* the condition range: the
# contract clips to 0.3 m near, this encoding to 0.1 m, so a re-encode of an
# already-written condition simply never uses the bottom of the scale.
DEPTH_VIDEO_NEAR_METRES = 0.1
DEPTH_VIDEO_FAR_METRES = 256.0

# The proxy R channel compresses a much deeper range into fewer codes, and
# reserves the top code as a sentinel so sky is distinguishable from "far".
#
# Note this channel runs *forward* — near is the low code — while the depth
# video beside it runs inverted. That looks like an inconsistency and is not:
# both place the sky sentinel immediately past the far plane. The depth video's
# sentinel is 0, so its valid codes have to descend towards far (near 255 ->
# far 1 -> sky 0); this channel's sentinel is 255, so its valid codes have to
# ascend towards far (near 0 -> far 254 -> sky 255). Inverting this one would
# put the nearest possible surface at 254 next to sky at 255, which is exactly
# the collision the sentinel exists to avoid.
PROXY_NEAR_METRES = 0.1
PROXY_FAR_METRES = 8000.0
PROXY_MAX_CODE = 254
PROXY_SKY_CODE = 255

# Per DATA_F.md's proxy table, keyed by the 11-class IDs it also defines. `ego`
# is not a class: it is `vehicle` in a clip whose protagonist is driving, which
# is why composing a proxy needs that flag as well as the labels.
_PROXY_GB_STANDARD11: dict[int, tuple[int, int]] = {
    tax.S11_SKY: (255, 255),
    tax.S11_PLAYER: (0, 255),
    tax.S11_PED: (0, 128),
    tax.S11_VEHICLE: (64, 0),
    tax.S11_BUILDING: (0, 0),
    tax.S11_ROAD: (255, 255),
    tax.S11_GROUND: (0, 0),
    tax.S11_VEGETATION: (255, 0),
    tax.S11_TERRAIN: (0, 0),
    tax.S11_WATER: (0, 0),
    tax.S11_PROP: (0, 0),
}
_PROXY_EGO_GB = (128, 0)

# coarse6 carries the same distinctions under different names, so it can be
# projected onto the table above rather than refused.
_COARSE6_TO_STANDARD11: dict[int, int] = {
    tax.C6_BACKGROUND: tax.S11_PROP,
    tax.C6_ROAD: tax.S11_ROAD,
    tax.C6_VEGETATION: tax.S11_VEGETATION,
    tax.C6_VEHICLE: tax.S11_VEHICLE,
    tax.C6_NPC: tax.S11_PED,
    tax.C6_HERO: tax.S11_PLAYER,
}


class EncodeError(RuntimeError):
    pass


def _log_code(
    metres: np.ndarray, *, near: float, far: float, top: int, inverted: bool
) -> np.ndarray:
    """Logarithmic quantisation of metric depth onto `0..top`."""
    span = math.log(far) - math.log(near)
    clipped = np.clip(metres.astype(np.float64), near, far)
    fraction = (math.log(far) - np.log(clipped)) / span
    if not inverted:
        fraction = 1.0 - fraction
    return np.rint(fraction * top).astype(np.int32)


def encode_depth_frame(metres: np.ndarray) -> np.ndarray:
    """One metric depth map as DATA_F.md's 8-bit grey log-z frame.

    Mirrors the document's two-step quantisation exactly, including the detour
    through 16 bits: going straight to 8 rounds differently near the far plane,
    and the decode formula published alongside it assumes this one.
    """
    metres = np.asarray(metres, dtype=np.float32)
    invalid = metres <= contract.DEPTH_VALID_EPSILON_METRES

    q16 = _log_code(
        metres,
        near=DEPTH_VIDEO_NEAR_METRES,
        far=DEPTH_VIDEO_FAR_METRES,
        top=65535,
        inverted=True,
    )
    grey = np.maximum(1, np.rint(q16 * 255.0 / 65535.0)).astype(np.uint8)
    grey[q16 == 0] = 0
    grey[invalid] = 0
    return grey


def decode_depth_frame(grey: np.ndarray) -> np.ndarray:
    """Inverse of `encode_depth_frame`; 0 becomes 0, meaning no depth."""
    grey = np.asarray(grey)
    span = math.log(DEPTH_VIDEO_FAR_METRES) - math.log(DEPTH_VIDEO_NEAR_METRES)
    metres = np.exp(math.log(DEPTH_VIDEO_FAR_METRES) - (grey / 255.0) * span)
    return np.where(grey == 0, 0.0, metres).astype(np.float32)


def encode_semantic_frame(ids: np.ndarray) -> np.ndarray:
    """One ID map as (R, G, B) = (0, 0, id), ready for a lossless RGB encode."""
    ids = np.asarray(ids, dtype=np.uint8)
    frame = np.zeros((*ids.shape, 3), dtype=np.uint8)
    frame[:, :, 2] = ids
    return frame


def to_standard11(ids: np.ndarray, taxonomy: str) -> np.ndarray:
    """Project a condition_root's labels onto the 11-class delivery IDs."""
    ids = np.asarray(ids, dtype=np.uint8)
    if taxonomy == "standard11":
        return ids
    if taxonomy == "coarse6":
        lut = np.zeros(256, dtype=np.uint8)
        for source, target in _COARSE6_TO_STANDARD11.items():
            lut[source] = target
        return lut[ids]
    if taxonomy == "cwm12":
        return tax.to_standard11(ids)
    raise EncodeError(f"cannot project taxonomy {taxonomy!r} onto the 11-class proxy table")


def compose_proxy_frame(
    metres: np.ndarray,
    ids_standard11: np.ndarray,
    *,
    driving: bool = False,
    inverted_depth: bool = False,
) -> np.ndarray:
    """Build one proxy frame: R is log-z, G and B carry the semantic colour.

    R is forward by default — near is code 0, far is 254, sky is 255. See the
    note on `PROXY_SKY_CODE` for why that is the only reading of DATA_F.md that
    keeps the sentinel usable, despite the depth video running the other way.
    """
    metres = np.asarray(metres, dtype=np.float32)
    ids_standard11 = np.asarray(ids_standard11, dtype=np.uint8)
    if metres.shape != ids_standard11.shape:
        raise EncodeError(f"depth {metres.shape} and labels {ids_standard11.shape} disagree")

    red = _log_code(
        metres,
        near=PROXY_NEAR_METRES,
        far=PROXY_FAR_METRES,
        top=PROXY_MAX_CODE,
        inverted=inverted_depth,
    ).astype(np.uint8)

    sky = (metres <= contract.DEPTH_VALID_EPSILON_METRES) | (ids_standard11 == tax.S11_SKY)
    red[sky] = PROXY_SKY_CODE

    green = np.zeros_like(red)
    blue = np.zeros_like(red)
    for class_id, (g, b) in _PROXY_GB_STANDARD11.items():
        if class_id == tax.S11_VEHICLE and driving:
            g, b = _PROXY_EGO_GB
        mask = ids_standard11 == class_id
        green[mask] = g
        blue[mask] = b

    return np.stack([red, green, blue], axis=-1)


@dataclass(frozen=True)
class _Encoder:
    process: subprocess.Popen
    path: Path

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait() != 0:
            stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise EncodeError(f"ffmpeg failed writing {self.path}: {stderr.strip()}")


# How each delivery stream must be encoded, as (input pixel format, codec,
# output pixel format). Only `color` is allowed to be lossy: it carries a
# photograph, where 8-bit YUV with chroma subsampling is the format everything
# downstream expects. The other three carry numbers pretending to be pixels, so
# they go out bit-exact - `libx264rgb` keeps RGB as RGB, where plain `libx264`
# would convert to YUV and subsample neighbouring IDs into classes that were
# never predicted.
_STREAM_FORMATS: dict[str, tuple[str, str, str]] = {
    "depth": ("gray", "libx264", "gray"),
    "semantic": ("rgb24", "libx264rgb", "rgb24"),
    "proxy": ("rgb24", "libx264rgb", "rgb24"),
    "color": ("rgb24", "libx264", "yuv420p"),
}

LOSSLESS_CRF = 0
DEFAULT_COLOR_CRF = 16


def open_encoder(
    path: Path, width: int, height: int, fps: float, *, kind: str, crf: int | None = None
) -> _Encoder:
    """Start an ffmpeg process writing one delivery stream, at any resolution."""
    if kind not in _STREAM_FORMATS:
        raise EncodeError(f"unknown stream kind {kind!r}; expected one of {sorted(_STREAM_FORMATS)}")
    in_pix_fmt, codec, out_pix_fmt = _STREAM_FORMATS[kind]
    if crf is None:
        crf = DEFAULT_COLOR_CRF if kind == "color" else LOSSLESS_CRF

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        in_pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        out_pix_fmt,
        str(path),
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise EncodeError("ffmpeg is not on PATH; it is required to write the delivery videos") from exc
    return _Encoder(process=process, path=path)


def write_videos(
    condition_root: Path,
    out_dir: Path | None = None,
    *,
    fps: float | None = None,
    kinds: tuple[str, ...] = ("depth", "semantic", "proxy"),
    inverted_proxy_depth: bool = False,
) -> dict:
    """Re-encode a written condition_root into the delivery videos.

    Reads back rather than keeping the stacks in memory, so this works on
    episodes far longer than one window and can be re-run on a condition_root
    that was extracted earlier.
    """
    condition_root = Path(condition_root)
    out_dir = Path(out_dir) if out_dir is not None else condition_root
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = condition_root / "extraction_report.json"
    if not report_path.is_file():
        raise EncodeError(f"no extraction_report.json under {condition_root}")
    report = json.loads(report_path.read_text())

    taxonomy = report.get("semantic", {}).get("taxonomy", "cwm12")
    driving = bool(report.get("semantic", {}).get("hero_split", {}).get("driving", False))
    frames = int(report["frames"])
    if fps is None:
        fps = _source_fps(report) or 24.0

    encoders: dict[str, _Encoder] = {}
    paths: dict[str, str] = {}
    for kind in kinds:
        path = out_dir / f"{kind}.mp4"
        encoders[kind] = open_encoder(
            path,
            contract.CONDITION_WIDTH,
            contract.CONDITION_HEIGHT,
            fps,
            kind=kind,
        )
        paths[kind] = str(path)

    try:
        for ordinal in range(frames):
            metres, ids = contract.read_frame(condition_root, ordinal)
            standard11 = to_standard11(ids, taxonomy) if "proxy" in encoders else None
            if "depth" in encoders:
                encoders["depth"].write(encode_depth_frame(metres))
            if "semantic" in encoders:
                encoders["semantic"].write(encode_semantic_frame(ids))
            if "proxy" in encoders:
                encoders["proxy"].write(
                    compose_proxy_frame(
                        metres,
                        standard11,
                        driving=driving,
                        inverted_depth=inverted_proxy_depth,
                    )
                )
    finally:
        errors = []
        for encoder in encoders.values():
            try:
                encoder.close()
            except EncodeError as exc:
                errors.append(str(exc))
        if errors:
            raise EncodeError("; ".join(errors))

    return {
        "condition_root": str(condition_root),
        "frames": frames,
        "fps": fps,
        "taxonomy": taxonomy,
        "driving": driving,
        "proxy_depth_inverted": inverted_proxy_depth,
        "videos": paths,
    }


def _source_fps(report: dict) -> float | None:
    """The source clip's frame rate, if the video it names is still reachable."""
    source = report.get("source_video")
    if not source or not Path(source).is_file():
        return None
    try:
        from .video import probe

        rate = probe(Path(source)).fps
    except Exception:  # noqa: BLE001 - fps is a nicety, never a reason to fail
        return None
    return rate if rate and rate > 0 else None
