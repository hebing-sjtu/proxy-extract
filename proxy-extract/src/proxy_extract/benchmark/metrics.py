"""Depth and semantic accuracy against ground truth.

Depth is scored twice on purpose: once as delivered, and once after removing
the single best global scale factor. A monocular metric model can be right
about the shape of a scene and wrong about how big it is, and those two failures
need completely different fixes — one wants a better model, the other wants a
scale anchor such as the COLMAP sparse cloud. A single AbsRel number hides which
one you have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthScore:
    abs_rel: float
    abs_rel_scaled: float
    delta1: float
    scale: float
    scale_drift: float
    valid_fraction: float

    @property
    def scale_limited(self) -> bool:
        """Is the error mostly a scale error rather than a geometry error?

        If fitting one number per clip removes most of the error, the model
        understands the scene and is simply miscalibrated — which is fixable
        without touching the model.
        """
        return self.abs_rel > 0 and self.abs_rel_scaled < 0.5 * self.abs_rel


def _valid(truth: np.ndarray, prediction: np.ndarray, sky_metres: float) -> np.ndarray:
    return (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & (truth > 0)
        & (prediction > 0)
        & (truth < sky_metres)
    )


def score_depth(
    truth: np.ndarray, prediction: np.ndarray, *, sky_metres: float = 256.0
) -> DepthScore:
    """Compare predicted metric depth against GT, `(frames, H, W)` both.

    Sky is excluded rather than counted as far: the GT writes it as the far
    plane, so scoring it would reward a model for predicting 256 m at the top
    of every frame and swamp the terms that matter.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"shape mismatch: truth {truth.shape} vs prediction {prediction.shape}")

    mask = _valid(truth, prediction, sky_metres)
    valid_fraction = float(mask.mean())
    if not mask.any():
        return DepthScore(float("nan"), float("nan"), 0.0, float("nan"), float("nan"), 0.0)

    t, p = truth[mask], prediction[mask]
    abs_rel = float(np.mean(np.abs(p - t) / t))

    # Median of the ratio, not least squares: depth error is multiplicative and
    # heavy-tailed, so a few near-camera pixels would otherwise set the scale.
    scale = float(np.median(t / p))
    abs_rel_scaled = float(np.mean(np.abs(p * scale - t) / t))

    ratio = np.maximum(p * scale / t, t / (p * scale))
    delta1 = float(np.mean(ratio < 1.25))

    return DepthScore(
        abs_rel=abs_rel,
        abs_rel_scaled=abs_rel_scaled,
        delta1=delta1,
        scale=scale,
        scale_drift=_scale_drift(truth, prediction, sky_metres),
        valid_fraction=valid_fraction,
    )


def _scale_drift(truth: np.ndarray, prediction: np.ndarray, sky_metres: float) -> float:
    """Spread of the per-frame scale factor across a clip, relative to its median.

    The failure mode specific to per-frame monocular prediction: each frame can
    be internally consistent while the clip breathes. Invisible to any
    per-frame metric, and fatal for anything that reads the clip as a sequence.
    """
    if truth.ndim != 3 or len(truth) < 2:
        return 0.0
    scales = []
    for frame_truth, frame_prediction in zip(truth, prediction):
        mask = _valid(frame_truth, frame_prediction, sky_metres)
        if mask.sum() > 16:
            scales.append(float(np.median(frame_truth[mask] / frame_prediction[mask])))
    if len(scales) < 2:
        return 0.0
    scales_array = np.asarray(scales)
    median = float(np.median(scales_array))
    if median == 0.0:
        return 0.0
    return float(np.std(scales_array) / abs(median))


# ---------------------------------------------------------------- semantic


def confusion(truth: np.ndarray, prediction: np.ndarray, num_classes: int) -> np.ndarray:
    """(num_classes, num_classes) counts, rows truth and columns prediction."""
    truth = np.asarray(truth).ravel()
    prediction = np.asarray(prediction).ravel()
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have the same number of pixels")
    valid = (truth < num_classes) & (prediction < num_classes)
    flat = np.bincount(
        truth[valid].astype(np.int64) * num_classes + prediction[valid].astype(np.int64),
        minlength=num_classes * num_classes,
    )
    return flat.reshape(num_classes, num_classes)


def iou_from_confusion(matrix: np.ndarray) -> np.ndarray:
    """Per-class IoU. Classes absent from both truth and prediction are NaN.

    NaN rather than zero because a class that never appears has no score, and
    averaging a zero in would punish a clip for the contents of its scene. This
    is exactly how `hero` ended up NaN in the earlier study — there the cause
    was a class nothing could predict, which is worth being able to see.
    """
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, np.nan)


def score_semantic(truth: np.ndarray, prediction: np.ndarray, num_classes: int) -> dict:
    matrix = confusion(truth, prediction, num_classes)
    iou = iou_from_confusion(matrix)
    present = matrix.sum(axis=1)
    return {
        "per_class_iou": [None if np.isnan(v) else round(float(v), 4) for v in iou],
        "miou": round(float(np.nanmean(iou)), 4) if np.any(~np.isnan(iou)) else None,
        "pixel_accuracy": round(float(np.diag(matrix).sum() / max(matrix.sum(), 1)), 4),
        "truth_pixel_share": [round(float(v), 5) for v in present / max(present.sum(), 1)],
    }


def confusion_pairs(matrix: np.ndarray, names: tuple[str, ...], *, top: int = 5) -> list[dict]:
    """The heaviest off-diagonal cells, as truth->prediction pairs.

    For the road/ground question specifically: the two are defined by material
    in the engine and by function in any image model, so they are expected to
    trade pixels. Seeing that pair at the top confirms the mismatch is the one
    already predicted rather than a mapping bug.
    """
    off_diagonal = matrix.astype(np.float64).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    total = max(off_diagonal.sum(), 1.0)

    pairs = []
    for index in np.argsort(off_diagonal, axis=None)[::-1][:top]:
        row, column = np.unravel_index(index, off_diagonal.shape)
        if off_diagonal[row, column] <= 0:
            break
        pairs.append(
            {
                "truth": names[row],
                "predicted": names[column],
                "pixels": int(off_diagonal[row, column]),
                "share_of_errors": round(float(off_diagonal[row, column] / total), 4),
            }
        )
    return pairs
