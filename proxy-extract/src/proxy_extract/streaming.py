"""Run the whole-episode stages without holding the whole episode.

`stabilize_pair` and `apply_range_guard` were both written to take a stack and
return a stack. On a 1800-frame episode at 1280x720 that is 6.6 GB of float32
depth plus its output, and it is why a delivery worker peaked around 40 GB and
only three of them fitted alongside a GPU - far too few to keep an H200 busy.

Neither stage actually needs the episode. The flow-compensated pass reads frame
`i` from `i-radius .. i+radius` and nothing else, so it runs in a sliding window
that is exact rather than approximate: a frame is emitted only once it has the
same neighbours it would have had in the batch call. The range guard needs one
global number, the median, which a histogram accumulates in constant space.

What is left genuinely global is the protagonist tracker, which has to see every
frame before it can say which person is the protagonist. That one keeps its
stack - but of labels only, a byte a pixel rather than four.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .temporal import DEFAULT_RADIUS, flow_compensated_pair

# Frames emitted per flow pass. The pass is re-run over `radius` frames of
# overlap on each side, so this trades a little repeated optical flow -
# 2*radius/(block + 2*radius), about 3% at the default - against a buffer that
# grows with it. Farneback is the most expensive thing on the CPU, so the
# overlap is not free, but a bigger block buys progressively less.
DEFAULT_BLOCK = 128


class WindowStabiliser:
    """`temporal.flow_compensated_pair` as a sliding window over a frame stream.

    Push frames in decode order; take stabilised frames out in the same order,
    delayed by `radius`. The output is identical to calling the batch function
    on the whole episode, because a frame is only released once every neighbour
    that call would have shown it is in the buffer.
    """

    def __init__(
        self,
        *,
        radius: int = DEFAULT_RADIUS,
        flow_downscale: int = 1,
        block: int = DEFAULT_BLOCK,
        flow_compensate: bool = True,
    ) -> None:
        if block < 1:
            raise ValueError(f"block must be >= 1, got {block}")
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        self.radius = radius
        self.block = block
        self.flow_downscale = flow_downscale
        self.flow_compensate = flow_compensate

        self._depth: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._guide: list[np.ndarray] = []
        # Absolute index of `self._depth[0]`, and of the next frame to release.
        self._base = 0
        self._emitted = 0

    @property
    def emitted(self) -> int:
        return self._emitted

    def push(
        self, depth: np.ndarray, labels: np.ndarray, guide: np.ndarray | None = None
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Accept one raw frame, yielding whichever earlier frames are now settled."""
        if self.flow_compensate and guide is None:
            raise ValueError("flow compensation is on, so every frame needs a guide")
        self._depth.append(np.asarray(depth, dtype=np.float32))
        self._labels.append(np.asarray(labels, dtype=np.uint8))
        if self.flow_compensate:
            self._guide.append(guide)
        if self._settled() >= self.block:
            yield from self._flush(final=False)

    def close(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Release the tail, whose right-hand neighbours do not exist."""
        yield from self._flush(final=True)

    def _settled(self) -> int:
        """Frames that could be released now with the full window behind them."""
        return max(self._base + len(self._depth) - self.radius - self._emitted, 0)

    def _flush(self, *, final: bool) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        start = self._emitted - self._base
        stop = len(self._depth) if final else len(self._depth) - self.radius
        if stop <= start:
            return

        steady_depth, steady_labels = flow_compensated_pair(
            np.stack(self._depth),
            np.stack(self._labels),
            guide_frames=list(self._guide) if self.flow_compensate else None,
            radius=self.radius,
            flow_downscale=self.flow_downscale,
        )
        for offset in range(start, stop):
            yield self._base + offset, steady_depth[offset], steady_labels[offset]

        self._emitted = self._base + stop
        self._trim()

    def _trim(self) -> None:
        """Drop frames no future window can reach."""
        keep_from = max(self._emitted - self.radius, 0)
        drop = keep_from - self._base
        if drop <= 0:
            return
        del self._depth[:drop]
        del self._labels[:drop]
        if self.flow_compensate:
            del self._guide[:drop]
        self._base = keep_from


# float16 has 65536 bit patterns, and for non-negative values their integer
# ordering is the same as their numeric ordering. So a count per pattern is an
# exact histogram of the depth this pipeline stores, not an approximation of it.
_FLOAT16_PATTERNS = 1 << 16


class RangeGuard:
    """`depth.scale.apply_range_guard`, one frame at a time.

    The clip is per-frame already. Only the reported median was global, and it
    is recovered exactly rather than estimated: depth is delivered as float16,
    so counting bit patterns tallies every value the stream can hold in 128 KB,
    whatever the episode's length.
    """

    def __init__(self, *, near: float, far: float) -> None:
        self.near = float(near)
        self.far = float(far)
        self._counts = np.zeros(_FLOAT16_PATTERNS, dtype=np.int64)
        self._valid = 0
        self._near_clipped = 0
        self._far_clipped = 0

    def apply(self, metres: np.ndarray) -> np.ndarray:
        """Record one frame's statistics and return it clipped to the range."""
        metres = np.asarray(metres, dtype=np.float32)
        valid = np.isfinite(metres) & (metres > 0)

        self._valid += int(valid.sum())
        self._near_clipped += int((valid & (metres < self.near)).sum())
        self._far_clipped += int((valid & (metres > self.far)).sum())
        kept = metres[valid].astype(np.float16).view(np.uint16)
        self._counts += np.bincount(kept, minlength=_FLOAT16_PATTERNS)

        return np.where(valid, np.clip(metres, self.near, self.far), 0.0).astype(np.float32)

    def stats(self) -> dict:
        total = max(self._valid, 1)
        return {
            "clipped_near_fraction": round(self._near_clipped / total, 6),
            "clipped_far_fraction": round(self._far_clipped / total, 6),
            "median_metres": round(self._median(), 4),
        }

    def state(self) -> dict:
        """Everything needed to carry the accumulator across a restart.

        The bit-pattern counts are the bulk of it and go out as a list; at
        65536 entries that is a few hundred kilobytes of JSON per checkpoint,
        which is nothing beside the frames written alongside it.
        """
        return {
            "valid": self._valid,
            "near_clipped": self._near_clipped,
            "far_clipped": self._far_clipped,
            "counts": np.flatnonzero(self._counts).tolist(),
            "weights": self._counts[np.flatnonzero(self._counts)].tolist(),
        }

    def restore(self, state: dict) -> None:
        self._valid = int(state["valid"])
        self._near_clipped = int(state["near_clipped"])
        self._far_clipped = int(state["far_clipped"])
        self._counts[:] = 0
        self._counts[np.asarray(state["counts"], dtype=np.int64)] = np.asarray(
            state["weights"], dtype=np.int64
        )

    def _median(self) -> float:
        if self._valid == 0:
            return 0.0
        # numpy's median of an even-length sample averages the middle two, so
        # match that rather than taking the lower of the pair.
        cumulative = np.cumsum(self._counts)
        lower = int(np.searchsorted(cumulative, (self._valid + 1) // 2))
        upper = int(np.searchsorted(cumulative, self._valid // 2 + 1))
        pair = np.array([lower, upper], dtype=np.uint16).view(np.float16)
        return float(pair.astype(np.float64).mean())


class FlickerMeter:
    """`temporal.flicker_rate` over a stream instead of a stack.

    The rate is a mean over consecutive pairs, so it needs the previous frame
    and two running totals - not the raw labels of an entire episode, which is
    the only reason those were ever kept once the stabiliser had consumed them.
    """

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None
        self._changed = 0
        self._compared = 0

    def push(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels)
        if self._previous is not None:
            self._changed += int((labels != self._previous).sum())
            self._compared += labels.size
        self._previous = labels

    @property
    def rate(self) -> float:
        if not self._compared:
            return 0.0
        return round(self._changed / self._compared, 6)

    def state(self) -> dict:
        return {"changed": self._changed, "compared": self._compared}

    def restore(self, state: dict) -> None:
        self._changed = int(state["changed"])
        self._compared = int(state["compared"])


def flicker_rate_of(labels: np.ndarray) -> float:
    """`temporal.flicker_rate` without its episode-sized temporary.

    The batch version compares two whole stacks, which allocates a boolean the
    size of the episode - 1.7 GB at delivery resolution, for one number.
    """
    meter = FlickerMeter()
    for frame in labels:
        meter.push(frame)
    return meter.rate


def class_fractions(labels: np.ndarray, names: dict[int, str] | list[str]) -> dict[str, float]:
    """What fraction of pixels each class holds, in one pass.

    Asking `(labels == cls).mean()` per class walks the episode once per class
    and builds an episode-sized boolean each time. Counting every class at once
    reads it once and holds a frame.
    """
    counts = np.zeros(256, dtype=np.int64)
    for frame in labels:
        counts += np.bincount(frame.ravel(), minlength=256)
    total = max(int(counts.sum()), 1)
    return {
        names[cls]: round(float(counts[cls]) / total, 6)
        for cls in np.flatnonzero(counts).tolist()
    }
