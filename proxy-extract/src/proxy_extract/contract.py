"""The `condition_root` contract consumed by code-world-model.

This module is the authoritative writer for that format. The numbers below are
duplicated from `cwm_h3_inference.constants` on purpose so this package does not
depend on the inference repo; `tests/test_contract_matches_cwm.py` cross-checks
them against the real thing whenever it is importable.

Layout, one pair per source-frame ordinal, flat:

    000000.depth.f32      headerless C-order little-endian float32, 192x336, metres
    000000.semantic_id.png  8-bit grayscale PNG, 336x192, values in [0, 11]
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np

CONDITION_WIDTH = 336
CONDITION_HEIGHT = 192
DEPTH_NEAR_METRES = 0.3
DEPTH_FAR_METRES = 256.0
DEPTH_VALID_EPSILON_METRES = 1.0e-3
NUM_SEMANTIC_CLASSES = 12

WINDOW_FRAMES = 124
STRIDE_FRAMES = 90

DEPTH_BYTES = CONDITION_WIDTH * CONDITION_HEIGHT * 4
_LOG_FAR = math.log(DEPTH_FAR_METRES)
_LOG_SPAN = _LOG_FAR - math.log(DEPTH_NEAR_METRES)


def window_count_for(frames: int) -> int:
    """How many 124-frame windows a clip of `frames` frames supports."""
    if frames < WINDOW_FRAMES:
        return 0
    return 1 + (frames - WINDOW_FRAMES) // STRIDE_FRAMES


def frames_for_windows(windows: int) -> int:
    """Frame count a `windows`-window run consumes: 124 + 90 * (n - 1)."""
    if windows < 1:
        raise ValueError("windows must be >= 1")
    return WINDOW_FRAMES + STRIDE_FRAMES * (windows - 1)


# ------------------------------------------------------------------ encoding


def encode_depth_codes(metric: np.ndarray) -> np.ndarray:
    """Metric depth in metres to the uint16 log codes the DUV encoder expects.

    Mirrors `cwm_h3_inference.duv.load_duv_frame`. The mapping is inverted and
    logarithmic: 0.3 m -> 65535, 256 m -> 0. Depth at or below 1 mm is invalid
    and encodes to 0, which is also what 256 m encodes to.
    """
    metric = np.asarray(metric, dtype=np.float64)
    valid = metric > DEPTH_VALID_EPSILON_METRES
    codes = np.zeros(metric.shape, dtype=np.uint16)
    if bool(np.any(valid)):
        clipped = np.clip(metric[valid], DEPTH_NEAR_METRES, DEPTH_FAR_METRES)
        normalized = (_LOG_FAR - np.log(clipped)) / _LOG_SPAN
        codes[valid] = np.floor(np.clip(normalized, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    return codes


def decode_depth_codes(codes: np.ndarray) -> np.ndarray:
    """Inverse of `encode_depth_codes`, for round-trip checks. Code 0 -> 0."""
    codes = np.asarray(codes, dtype=np.float64)
    metric = np.exp(_LOG_FAR - (codes / 65535.0) * _LOG_SPAN)
    return np.where(codes > 0, metric, 0.0).astype(np.float32)


def depth_code_shift_for_scale(scale: float) -> float:
    """Code offset produced by multiplying every depth by `scale`.

    The encoding is logarithmic, so a global scale error is a uniform shift
    rather than a distortion. Useful for reasoning about calibration error
    budgets: a 2x scale error moves every code by ~6729 of 65535 (10.3%).
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return -math.log(scale) / _LOG_SPAN * 65535.0


# --------------------------------------------------------------- resampling


def _block_reduce(image: np.ndarray, out_h: int, out_w: int, reducer) -> np.ndarray:
    h, w = image.shape[:2]
    if h % out_h or w % out_w:
        raise ValueError(f"{h}x{w} is not an integer multiple of {out_h}x{out_w}")
    fh, fw = h // out_h, w // out_w
    tiles = image.reshape(out_h, fh, out_w, fw).swapaxes(1, 2).reshape(out_h, out_w, fh * fw)
    return reducer(tiles)


def downsample_depth(metric: np.ndarray, *, mode: str = "median") -> np.ndarray:
    """Reduce a metric depth map to the 192x336 condition grid.

    Averaging across a depth discontinuity invents surfaces that exist in
    neither the foreground nor the background ("flying pixels"), which at this
    aggressive a reduction is a real artifact rather than a nuisance. `median`
    picks a depth that actually occurred in the block; `min` biases to the
    foreground, which keeps thin near objects like railings.
    """
    metric = np.asarray(metric, dtype=np.float32)
    if metric.shape == (CONDITION_HEIGHT, CONDITION_WIDTH):
        return metric

    invalid = metric <= DEPTH_VALID_EPSILON_METRES
    work = np.where(invalid, np.nan, metric).astype(np.float32)

    if metric.shape[0] % CONDITION_HEIGHT or metric.shape[1] % CONDITION_WIDTH:
        import cv2

        work = cv2.resize(work, (CONDITION_WIDTH, CONDITION_HEIGHT), interpolation=cv2.INTER_NEAREST)
        return np.nan_to_num(work, nan=0.0).astype(np.float32)

    if mode == "median":
        reducer = lambda t: np.nanmedian(t, axis=2)  # noqa: E731
    elif mode == "min":
        reducer = lambda t: np.nanmin(t, axis=2)  # noqa: E731
    elif mode == "mean":
        reducer = lambda t: np.nanmean(t, axis=2)  # noqa: E731
    else:
        raise ValueError(f"unknown depth downsample mode: {mode}")

    # A block that is entirely invalid reduces to NaN, which is the expected
    # answer here (it becomes 0, i.e. "no depth"), not a condition to warn about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = _block_reduce(work, CONDITION_HEIGHT, CONDITION_WIDTH, reducer)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def downsample_semantic(ids: np.ndarray) -> np.ndarray:
    """Reduce a class-ID map to 192x336 by per-block majority vote.

    Nearest-neighbour sampling would keep or drop thin structures (railings,
    lamp posts, distant humans) depending on where the sample grid happens to
    land. A majority vote is stable and keeps whichever class actually
    dominates the block.
    """
    ids = np.asarray(ids)
    if ids.shape == (CONDITION_HEIGHT, CONDITION_WIDTH):
        return ids.astype(np.uint8)

    if ids.shape[0] % CONDITION_HEIGHT or ids.shape[1] % CONDITION_WIDTH:
        import cv2

        return cv2.resize(
            ids.astype(np.uint8), (CONDITION_WIDTH, CONDITION_HEIGHT), interpolation=cv2.INTER_NEAREST
        )

    def _mode(tiles: np.ndarray) -> np.ndarray:
        counts = np.zeros((*tiles.shape[:2], NUM_SEMANTIC_CLASSES), dtype=np.int32)
        for cls in range(NUM_SEMANTIC_CLASSES):
            counts[:, :, cls] = (tiles == cls).sum(axis=2)
        return counts.argmax(axis=2)

    return _block_reduce(ids.astype(np.uint8), CONDITION_HEIGHT, CONDITION_WIDTH, _mode).astype(np.uint8)


# ------------------------------------------------------------------- writing


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def frame_paths(root: Path, ordinal: int) -> tuple[Path, Path]:
    return root / f"{ordinal:06d}.depth.f32", root / f"{ordinal:06d}.semantic_id.png"


def write_frame(root: Path, ordinal: int, depth_metres: np.ndarray, semantic_ids: np.ndarray) -> None:
    """Write one ordinal's depth + semantic pair, resampling if needed."""
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    depth = downsample_depth(depth_metres)
    semantic = downsample_semantic(semantic_ids)

    if not np.all(np.isfinite(depth)):
        raise ValueError(f"depth for ordinal {ordinal} contains non-finite values")
    depth = np.clip(depth, 0.0, None)
    if int(semantic.max(initial=0)) >= NUM_SEMANTIC_CLASSES:
        raise ValueError(f"semantic ids for ordinal {ordinal} exceed {NUM_SEMANTIC_CLASSES - 1}")

    depth_path, semantic_path = frame_paths(root, ordinal)
    _atomic_write(depth_path, np.ascontiguousarray(depth, dtype="<f4").tobytes(order="C"))

    tmp = semantic_path.with_suffix(".png.tmp")
    Image.fromarray(semantic.astype(np.uint8), mode="L").save(tmp, format="PNG", optimize=True)
    os.replace(tmp, semantic_path)


def read_frame(root: Path, ordinal: int) -> tuple[np.ndarray, np.ndarray]:
    """Read back one ordinal as (metric depth, semantic ids)."""
    from PIL import Image

    depth_path, semantic_path = frame_paths(root, ordinal)
    payload = depth_path.read_bytes()
    if len(payload) != DEPTH_BYTES:
        raise ValueError(f"raw depth byte count must be {DEPTH_BYTES}, got {len(payload)}: {depth_path}")
    depth = np.frombuffer(payload, dtype="<f4").reshape(CONDITION_HEIGHT, CONDITION_WIDTH)

    with Image.open(semantic_path) as image:
        if image.mode != "L" or image.size != (CONDITION_WIDTH, CONDITION_HEIGHT):
            raise ValueError(f"semantic ID must be L/{CONDITION_WIDTH}x{CONDITION_HEIGHT}: {semantic_path}")
        semantic = np.asarray(image, dtype=np.uint8).copy()
    return depth, semantic


# ---------------------------------------------------------------- validation


class ContractError(ValueError):
    """A condition_root does not satisfy what code-world-model will accept."""


def validate_condition_root(root: Path, *, expected_frames: int | None = None) -> dict:
    """Re-read a written condition_root and apply every check the loader applies.

    Running this in-process after writing is much cheaper than discovering a
    malformed frame when `prepare` throws 124 frames into a window.
    """
    root = Path(root)
    if not root.is_dir():
        raise ContractError(f"condition_root is not a directory: {root}")

    ordinals = sorted(int(p.name[:6]) for p in root.glob("??????.depth.f32"))
    if not ordinals:
        raise ContractError(f"no depth frames in {root}")
    if ordinals != list(range(len(ordinals))):
        raise ContractError(f"ordinals must be contiguous from 000000 in {root}")
    if expected_frames is not None and len(ordinals) != expected_frames:
        raise ContractError(f"expected {expected_frames} frames, found {len(ordinals)} in {root}")

    classes_seen: set[int] = set()
    depth_min, depth_max, invalid_total = math.inf, 0.0, 0
    for ordinal in ordinals:
        depth, semantic = read_frame(root, ordinal)
        if not bool(np.all(np.isfinite(depth))):
            raise ContractError(f"non-finite depth at ordinal {ordinal}")
        if bool(np.any(depth < 0.0)):
            raise ContractError(f"negative depth at ordinal {ordinal}")
        top = int(semantic.max(initial=0))
        if top >= NUM_SEMANTIC_CLASSES:
            raise ContractError(f"semantic id {top} out of range at ordinal {ordinal}")

        valid = depth > DEPTH_VALID_EPSILON_METRES
        invalid_total += int((~valid).sum())
        if bool(np.any(valid)):
            depth_min = min(depth_min, float(depth[valid].min()))
            depth_max = max(depth_max, float(depth[valid].max()))
        classes_seen.update(np.unique(semantic).tolist())

    pixels = len(ordinals) * CONDITION_HEIGHT * CONDITION_WIDTH
    return {
        "frames": len(ordinals),
        "windows": window_count_for(len(ordinals)),
        "depth_min_metres": None if depth_min is math.inf else round(depth_min, 4),
        "depth_max_metres": round(depth_max, 4),
        "invalid_depth_fraction": round(invalid_total / pixels, 6),
        "semantic_classes_present": sorted(classes_seen),
    }
