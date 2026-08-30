"""Turn up-to-scale depth into metres.

The DUV encoder is logarithmic, so a residual global scale error `s` shifts every
depth code by the same constant rather than warping the geometry - a 2x error
costs about 10.3% of full range. That makes a single well-estimated global scale
per clip sufficient, and makes per-frame absolute metric prediction unnecessary.

Preferred source of truth is the GT camera track: the ratio between the real and
the predicted camera baselines is exactly the scale factor, and it needs no
assumptions about scene content.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this, the camera barely moved and its baseline says nothing about scale.
MIN_BASELINE_METRES = 0.05
# Ratios spread wider than this suggest the predicted trajectory is not a
# similarity transform of the real one, i.e. the pose estimate is unreliable.
MAX_RATIO_DISPERSION = 0.25


@dataclass(frozen=True)
class ScaleSolution:
    scale: float
    solved: bool
    method: str
    dispersion: float | None = None
    reason: str | None = None


def solve_scale_from_cameras(
    predicted_positions: np.ndarray,
    gt_positions: np.ndarray,
    *,
    min_baseline: float = MIN_BASELINE_METRES,
    max_dispersion: float = MAX_RATIO_DISPERSION,
) -> ScaleSolution:
    """Scale factor mapping a predicted camera trajectory onto the real one.

    Compares pairwise camera-to-camera distances, which is invariant to the
    unknown rotation and translation between the two world frames, so no
    Procrustes alignment is needed. The median ratio is robust to a handful of
    badly predicted poses.
    """
    predicted = np.asarray(predicted_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"trajectory shapes differ: {predicted.shape} vs {gt.shape}")
    if len(predicted) < 2:
        return ScaleSolution(1.0, False, "cameras", reason="need at least two frames")

    iu = np.triu_indices(len(predicted), k=1)
    d_pred = np.linalg.norm(predicted[iu[0]] - predicted[iu[1]], axis=1)
    d_gt = np.linalg.norm(gt[iu[0]] - gt[iu[1]], axis=1)

    usable = (d_gt > min_baseline) & (d_pred > 1e-9)
    if int(usable.sum()) < max(3, len(predicted) // 10):
        return ScaleSolution(
            1.0, False, "cameras", reason=f"camera moved less than {min_baseline} m; scale unobservable"
        )

    ratios = d_gt[usable] / d_pred[usable]
    scale = float(np.median(ratios))
    dispersion = float(np.median(np.abs(ratios - scale)) / max(scale, 1e-9))
    if dispersion > max_dispersion:
        return ScaleSolution(
            scale, False, "cameras", dispersion, f"ratio dispersion {dispersion:.3f} exceeds {max_dispersion}"
        )
    return ScaleSolution(scale, True, "cameras", dispersion)


def solve_scale_shift(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    fit_shift: bool = False,
) -> ScaleSolution:
    """Least-squares fit of `source` onto a metric `reference` depth map.

    The fallback when no GT camera track is available: align an up-to-scale
    prediction against a per-frame metric model. Fitting in log space would be
    the natural choice for a pure scale, but a shift term only makes sense in
    linear space, so both live here and the caller picks.
    """
    source = np.asarray(source, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if source.shape != reference.shape:
        raise ValueError(f"shape mismatch: {source.shape} vs {reference.shape}")

    keep = np.isfinite(source) & np.isfinite(reference) & (source > 0) & (reference > 0)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    if int(keep.sum()) < 100:
        return ScaleSolution(1.0, False, "scale_shift", reason="fewer than 100 comparable pixels")

    src, ref = source[keep], reference[keep]
    if not fit_shift:
        # Median of per-pixel ratios in log space: robust and shift-free.
        scale = float(np.exp(np.median(np.log(ref) - np.log(src))))
        dispersion = float(np.median(np.abs(np.log(ref / (src * scale)))))
        return ScaleSolution(scale, dispersion < 0.5, "scale_shift", dispersion)

    design = np.stack([src, np.ones_like(src)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, ref, rcond=None)
    residual = float(np.median(np.abs(ref - (src * scale + shift))) / max(np.median(ref), 1e-9))
    solution = ScaleSolution(float(scale), scale > 0 and residual < 0.5, "scale_shift_affine", residual)
    return solution


def apply_range_guard(depth: np.ndarray, *, near: float, far: float) -> tuple[np.ndarray, dict]:
    """Report how much of a depth map the encoder's 0.3-256 m range will clip.

    Clipping is not an error - sky legitimately sits beyond the far plane - but
    a clip fraction that jumps between clips usually means the scale solve went
    wrong, so it is worth surfacing rather than silently clamping.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    total = max(int(valid.sum()), 1)
    stats = {
        "clipped_near_fraction": round(float((valid & (depth < near)).sum()) / total, 6),
        "clipped_far_fraction": round(float((valid & (depth > far)).sum()) / total, 6),
        "median_metres": round(float(np.median(depth[valid])) if valid.any() else 0.0, 4),
    }
    return np.where(valid, np.clip(depth, near, far), 0.0).astype(np.float32), stats
