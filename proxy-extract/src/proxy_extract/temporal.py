"""Temporal stabilisation of per-frame predictions.

Per-frame segmentation flips ambiguous regions between plausible classes from
one frame to the next, and per-frame depth wobbles. Fed into a video model as
conditioning, that flicker is a signal: it tells the model the world is
unstable. Suppressing it matters more here than squeezing out per-frame accuracy.

Voting across a temporal window naively assumes a pixel looks at the same
surface in every frame, which a moving camera breaks - at these clips' measured
~2.8 px/frame it smears edges by several pixels over a 5-frame window. So the
neighbours are warped into the current frame with optical flow before voting.
"""

from __future__ import annotations

import warnings

import numpy as np

from .taxonomy import NUM_CLASSES

DEFAULT_RADIUS = 2
# Suppress runs of a single frame. Raising this trades responsiveness to genuine
# fast changes for more aggressive flicker removal.
DEFAULT_MIN_RUN = 2


def _flow_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(source, target, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp(image: np.ndarray, flow: np.ndarray, *, nearest: bool) -> np.ndarray:
    """Pull `image` along `flow` into the reference frame's pixel grid."""
    import cv2

    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        image,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _gray_stack(frames: list[np.ndarray], shape: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    out = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        if gray.shape != shape:
            gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        out.append(gray)
    return out


def _neighbour_flows(guide: list[np.ndarray], index: int, offsets: range) -> dict[int, np.ndarray]:
    flows: dict[int, np.ndarray] = {}
    for offset in offsets:
        other = index + offset
        if offset == 0 or not 0 <= other < len(guide):
            continue
        flows[other] = _flow_between(guide[index], guide[other])
    return flows


def suppress_short_runs(labels: np.ndarray, *, min_run: int = DEFAULT_MIN_RUN) -> np.ndarray:
    """Erase label runs shorter than `min_run` frames along the time axis.

    A majority vote cannot fix the worst case on its own. Perfect frame-to-frame
    alternation is a root signal of any odd-length mode or median filter: the
    window centred on a pixel always contains one more copy of that pixel's own
    class than of the other, so the vote re-elects the flicker. Collapsing short
    runs attacks it directly, and a genuine sustained change - a run longer than
    `min_run` - passes through untouched.
    """
    labels = np.asarray(labels, dtype=np.uint8).copy()
    frames = labels.shape[0]
    if min_run < 2 or frames < 3:
        return labels

    # Run length at every (frame, pixel): how far the current label extends
    # backwards, plus how far it extends forwards. Two sequential passes over
    # the time axis, each vectorised across all pixels.
    backward = np.ones_like(labels, dtype=np.int32)
    for t in range(1, frames):
        backward[t] = np.where(labels[t] == labels[t - 1], backward[t - 1] + 1, 1)
    forward = np.ones_like(labels, dtype=np.int32)
    for t in range(frames - 2, -1, -1):
        forward[t] = np.where(labels[t] == labels[t + 1], forward[t + 1] + 1, 1)
    too_short = (backward + forward - 1) < min_run

    # Short runs adopt the label of whatever precedes them. A short run at the
    # very start has nothing before it, so seed those from the future first.
    for t in range(frames - 2, -1, -1):
        labels[t] = np.where(too_short[t], labels[t + 1], labels[t])
    for t in range(1, frames):
        labels[t] = np.where(too_short[t], labels[t - 1], labels[t])
    return labels


def stabilize_labels(
    labels: np.ndarray,
    *,
    guide_frames: list[np.ndarray] | None = None,
    radius: int = DEFAULT_RADIUS,
    min_run: int = DEFAULT_MIN_RUN,
) -> np.ndarray:
    """Flow-compensated majority vote, then short-run suppression.

    The two passes fix different failures. The vote cleans up spatially noisy
    boundaries and one-off misclassifications; the run filter removes the
    periodic alternation the vote provably cannot. The current frame carries
    weight 2 in the vote so a sustained change wins immediately rather than
    lagging by half the window.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape[0] < 3:
        return labels
    if radius < 1:
        return suppress_short_runs(labels, min_run=min_run)

    shape = labels.shape[1:]
    guide = _gray_stack(guide_frames, shape) if guide_frames is not None else None
    if guide is not None and len(guide) != len(labels):
        raise ValueError(f"guide has {len(guide)} frames for {len(labels)} label maps")

    offsets = range(-radius, radius + 1)
    out = np.empty_like(labels)
    for index in range(len(labels)):
        votes = np.zeros((*shape, NUM_CLASSES), dtype=np.int16)
        flows = _neighbour_flows(guide, index, offsets) if guide is not None else {}
        for offset in offsets:
            other = index + offset
            if not 0 <= other < len(labels):
                continue
            patch = labels[other]
            if other in flows:
                patch = _warp(patch, flows[other], nearest=True)
            weight = 2 if offset == 0 else 1
            # Accumulate class-by-class rather than scattering into `votes` with
            # fancy indexing: 12 vectorised comparisons beat a per-pixel
            # np.add.at by orders of magnitude at this frame count.
            for cls in range(NUM_CLASSES):
                votes[..., cls] += weight * (patch == cls)
        out[index] = votes.argmax(axis=2).astype(np.uint8)
    return suppress_short_runs(out, min_run=min_run)


def stabilize_depth(
    depth: np.ndarray,
    *,
    guide_frames: list[np.ndarray] | None = None,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    """Temporal median over a +/-`radius` window, flow-compensated.

    Median rather than mean: an occlusion boundary that a neighbouring frame
    disagrees about should be rejected outright, not averaged into a surface
    that exists at neither depth.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if radius < 1 or depth.shape[0] < 3:
        return depth

    shape = depth.shape[1:]
    guide = _gray_stack(guide_frames, shape) if guide_frames is not None else None
    if guide is not None and len(guide) != len(depth):
        raise ValueError(f"guide has {len(guide)} frames for {len(depth)} depth maps")

    offsets = range(-radius, radius + 1)
    out = np.empty_like(depth)
    for index in range(len(depth)):
        flows = _neighbour_flows(guide, index, offsets) if guide is not None else {}
        stack = []
        for offset in offsets:
            other = index + offset
            if not 0 <= other < len(depth):
                continue
            patch = depth[other]
            if other in flows:
                patch = _warp(patch, flows[other], nearest=False)
            stack.append(np.where(patch > 0, patch, np.nan))
        # A pixel invalid across the whole window reduces to NaN, which is the
        # right answer (it becomes 0), not something to warn about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            merged = np.nanmedian(np.stack(stack), axis=0)
        out[index] = np.nan_to_num(merged, nan=0.0).astype(np.float32)
    return out


def flicker_rate(labels: np.ndarray) -> float:
    """Fraction of pixels whose class changes between consecutive frames.

    A single number for whether stabilisation helped, comparable across clips.
    """
    labels = np.asarray(labels)
    if len(labels) < 2:
        return 0.0
    changes = labels[1:] != labels[:-1]
    return round(float(changes.mean()), 6)
