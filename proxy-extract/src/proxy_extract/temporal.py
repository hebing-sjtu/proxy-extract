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
# `suppress_short_runs` saturates its run counters into a uint8 to keep a
# whole-episode buffer affordable, so it cannot represent a threshold a byte
# cannot hold. Nothing near this is a sensible flicker setting anyway.
MAX_MIN_RUN = 255


def _flow_between(source: np.ndarray, target: np.ndarray, *, downscale: int = 1) -> np.ndarray:
    """Dense flow from `source` to `target`, optionally solved smaller.

    Farneback is quadratic in pixel count, and at 1280x720 a five-frame window
    costs four flows per frame, which for a 1800-frame episode dominates
    everything else on the CPU. Solving at 1/k and scaling the field back up
    costs k^2 less. Flow fields are smooth over most of the image, so the error
    concentrates at motion boundaries, which is also where stabilisation matters
    most - hence a parameter rather than a silent default.
    """
    import cv2

    if downscale < 1:
        raise ValueError(f"downscale must be >= 1, got {downscale}")
    if downscale == 1:
        return cv2.calcOpticalFlowFarneback(source, target, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    height, width = source.shape[:2]
    small = (max(width // downscale, 16), max(height // downscale, 16))
    flow = cv2.calcOpticalFlowFarneback(
        cv2.resize(source, small, interpolation=cv2.INTER_AREA),
        cv2.resize(target, small, interpolation=cv2.INTER_AREA),
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    # Displacements are in pixels, so they scale with the grid they were solved on.
    flow[..., 0] *= width / small[0]
    flow[..., 1] *= height / small[1]
    return flow


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


def _neighbour_flows(
    guide: list[np.ndarray], index: int, offsets: range, *, downscale: int = 1
) -> dict[int, np.ndarray]:
    flows: dict[int, np.ndarray] = {}
    for offset in offsets:
        other = index + offset
        if offset == 0 or not 0 <= other < len(guide):
            continue
        flows[other] = _flow_between(guide[index], guide[other], downscale=downscale)
    return flows


def _vote_at(
    labels: np.ndarray, index: int, offsets: range, flows: dict[int, np.ndarray]
) -> np.ndarray:
    """The weighted majority vote for one frame, given its neighbours' flows."""
    votes = np.zeros((*labels.shape[1:], NUM_CLASSES), dtype=np.int16)
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
    return votes.argmax(axis=2).astype(np.uint8)


def _median_at(
    depth: np.ndarray, index: int, offsets: range, flows: dict[int, np.ndarray]
) -> np.ndarray:
    """The temporal median for one frame, given its neighbours' flows."""
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
    return np.nan_to_num(merged, nan=0.0).astype(np.float32)


def suppress_short_runs(
    labels: np.ndarray, *, min_run: int = DEFAULT_MIN_RUN, in_place: bool = False
) -> np.ndarray:
    """Erase label runs shorter than `min_run` frames along the time axis.

    A majority vote cannot fix the worst case on its own. Perfect frame-to-frame
    alternation is a root signal of any odd-length mode or median filter: the
    window centred on a pixel always contains one more copy of that pixel's own
    class than of the other, so the vote re-elects the flicker. Collapsing short
    runs attacks it directly, and a genuine sustained change - a run longer than
    `min_run` - passes through untouched.

    `in_place` overwrites the caller's array. A whole episode of labels is 1.7
    GB, so the defensive copy is worth declining when the caller has just read
    the stack for this and nothing else.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    if not in_place:
        labels = labels.copy()
    frames = labels.shape[0]
    if min_run < 2 or frames < 3:
        return labels

    if min_run > MAX_MIN_RUN:
        raise ValueError(f"min_run must be at most {MAX_MIN_RUN}, got {min_run}")

    # Run length at every (frame, pixel): how far the current label extends
    # backwards, plus how far it extends forwards. The only question asked of
    # either count is whether it reaches `min_run`, so both saturate there and
    # fit in a byte.
    #
    # That matters because at the 1280x720 delivery size a whole-episode counter
    # is the largest array in the process. Holding two of them as int16 cost
    # 6.6 GB for a 1800-frame episode, which was most of the reason a worker
    # needed tens of gigabytes and only a few would fit alongside a GPU.
    #
    # One buffer serves for both: it is filled with the capped forward count,
    # then overwritten in place with the verdict, computed against a backward
    # count that never needs more than the previous frame.
    short = np.ones_like(labels, dtype=np.uint8)
    for t in range(frames - 2, -1, -1):
        grown = np.minimum(short[t + 1].astype(np.int16) + 1, min_run)
        short[t] = np.where(labels[t] == labels[t + 1], grown, 1)

    backward = np.ones(labels.shape[1:], dtype=np.uint8)
    for t in range(frames):
        if t:
            grown = np.minimum(backward.astype(np.int16) + 1, min_run)
            backward = np.where(labels[t] == labels[t - 1], grown, 1).astype(np.uint8)
        short[t] = (backward.astype(np.int16) + short[t] - 1) < min_run

    # Short runs adopt the label of whatever precedes them. A short run at the
    # very start has nothing before it, so seed those from the future first.
    for t in range(frames - 2, -1, -1):
        labels[t] = np.where(short[t], labels[t + 1], labels[t])
    for t in range(1, frames):
        labels[t] = np.where(short[t], labels[t - 1], labels[t])
    return labels


def stabilize_labels(
    labels: np.ndarray,
    *,
    guide_frames: list[np.ndarray] | None = None,
    radius: int = DEFAULT_RADIUS,
    min_run: int = DEFAULT_MIN_RUN,
    flow_downscale: int = 1,
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
        flows = (
            _neighbour_flows(guide, index, offsets, downscale=flow_downscale)
            if guide is not None
            else {}
        )
        out[index] = _vote_at(labels, index, offsets, flows)
    return suppress_short_runs(out, min_run=min_run)


def stabilize_depth(
    depth: np.ndarray,
    *,
    guide_frames: list[np.ndarray] | None = None,
    radius: int = DEFAULT_RADIUS,
    flow_downscale: int = 1,
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
        flows = (
            _neighbour_flows(guide, index, offsets, downscale=flow_downscale)
            if guide is not None
            else {}
        )
        out[index] = _median_at(depth, index, offsets, flows)
    return out


def stabilize_pair(
    depth: np.ndarray,
    labels: np.ndarray,
    *,
    guide_frames: list[np.ndarray] | None = None,
    radius: int = DEFAULT_RADIUS,
    min_run: int = DEFAULT_MIN_RUN,
    flow_downscale: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stabilise both stacks in one pass, solving each frame's flow once.

    The two single-stack functions ask `_neighbour_flows` for byte-identical
    fields — same guide, same offsets, same downscale — and Farneback is 95% of
    what a pass costs, so running them back to back paid for every flow twice.
    Sharing them changes nothing about the result; the warps differ (nearest for
    labels, linear for depth) but they are cheap and still done separately.
    """
    depth = np.asarray(depth, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    if len(depth) != len(labels):
        raise ValueError(f"{len(depth)} depth maps against {len(labels)} label maps")

    # The two disagree about what to do in the degenerate cases — a short stack
    # still gets its runs suppressed, for instance — so let each answer for
    # itself rather than restating the rules here and drifting from them.
    if len(depth) < 3 or radius < 1:
        return (
            stabilize_depth(
                depth, guide_frames=guide_frames, radius=radius, flow_downscale=flow_downscale
            ),
            stabilize_labels(
                labels,
                guide_frames=guide_frames,
                radius=radius,
                min_run=min_run,
                flow_downscale=flow_downscale,
            ),
        )

    steady_depth, steady_labels = flow_compensated_pair(
        depth,
        labels,
        guide_frames=guide_frames,
        radius=radius,
        flow_downscale=flow_downscale,
    )
    return steady_depth, suppress_short_runs(steady_labels, min_run=min_run)


def flow_compensated_pair(
    depth: np.ndarray,
    labels: np.ndarray,
    *,
    guide_frames: list[np.ndarray] | None = None,
    radius: int = DEFAULT_RADIUS,
    flow_downscale: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """The window pass alone: median for depth, vote for labels, no run filter.

    Split out from `stabilize_pair` because the two halves need different
    amounts of context. This one is strictly local - frame `i` reads only
    `i-radius .. i+radius` - which is what lets a streaming caller hold a
    handful of frames instead of the episode. Run suppression is not local in
    the same way, so it gets its own, wider window rather than forcing this one
    to be conservative for both.
    """
    depth = np.asarray(depth, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    shape = depth.shape[1:]
    if labels.shape[1:] != shape:
        raise ValueError(f"depth is {shape} but labels are {labels.shape[1:]}")
    guide = _gray_stack(guide_frames, shape) if guide_frames is not None else None
    if guide is not None and len(guide) != len(depth):
        raise ValueError(f"guide has {len(guide)} frames for {len(depth)} depth maps")

    offsets = range(-radius, radius + 1)
    steady_depth = np.empty_like(depth)
    steady_labels = np.empty_like(labels)
    for index in range(len(depth)):
        flows = (
            _neighbour_flows(guide, index, offsets, downscale=flow_downscale)
            if guide is not None
            else {}
        )
        steady_depth[index] = _median_at(depth, index, offsets, flows)
        steady_labels[index] = _vote_at(labels, index, offsets, flows)
    return steady_depth, steady_labels


def flicker_rate(labels: np.ndarray) -> float:
    """Fraction of pixels whose class changes between consecutive frames.

    A single number for whether stabilisation helped, comparable across clips.
    """
    labels = np.asarray(labels)
    if len(labels) < 2:
        return 0.0
    changes = labels[1:] != labels[:-1]
    return round(float(changes.mean()), 6)
