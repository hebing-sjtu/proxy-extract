"""The one-second grid every timestamped caption is written against.

Everything else in this package is downstream of one decision: **time is the
primary key, and a bin index is a name for a real interval of seconds**. The
older `prompt.json` numbered its chunks `0..3` and left the reader to divide
the duration by four, which makes the same index mean 1.29 s in a 5.17 s clip
and 15 s in a minute-long one. Nothing that trains on both can learn what a
chunk is worth.

A grid is fixed-width in seconds and therefore comparable across clips of any
length, and it is what makes the contract **closed under slicing**: cutting a
60 s caption down to the 5 s a training window actually shows is an integer
range of bins, not a re-annotation.

The tail is the only interesting case. 124 frames at 24 fps is 5.1667 s, so a
1 s grid leaves 0.1667 s over - four frames, which is not an observation, it is
a rounding error with a caption attached. `plan` folds a tail shorter than
`min_tail_seconds` into the bin before it, so that clip gets five bins and the
last one is 1.1667 s long rather than six bins where the last is noise. The
tail is never silently dropped and never silently rounded up: the bin carries
its real `start` and `stop`, and a reader that cares can see the width.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BIN_SECONDS = 1.0

# Below this a bin holds too few frames to say anything a captioner could have
# observed, so it joins its neighbour instead of becoming its own claim.
DEFAULT_MIN_TAIL_SECONDS = 0.5

# Times are seconds rounded here and nowhere else, so that a value written to
# JSON, read back, and compared is the same value. Milliseconds are finer than
# any frame rate this corpus uses.
TIME_PLACES = 3


def _round(seconds: float) -> float:
    return round(float(seconds), TIME_PLACES)


@dataclass(frozen=True)
class Bin:
    """One captioned interval, in both seconds and frames.

    `stop` and `stop_frame` are exclusive. Both representations are stored
    because the caption is written against seconds and the evidence is read
    out of frames, and re-deriving one from the other at each use is how the
    two drift apart by a frame at the seams.
    """

    index: int
    start: float
    stop: float
    first_frame: int
    stop_frame: int

    @property
    def seconds(self) -> float:
        return _round(self.stop - self.start)

    @property
    def frames(self) -> int:
        return self.stop_frame - self.first_frame

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "t": [self.start, self.stop],
            "frames": [self.first_frame, self.stop_frame],
        }


@dataclass(frozen=True)
class Grid:
    """The complete bin plan for one clip."""

    fps: float
    frames: int
    bin_seconds: float
    bins: tuple[Bin, ...]

    @property
    def duration(self) -> float:
        return _round(self.frames / self.fps)

    @property
    def count(self) -> int:
        return len(self.bins)

    def indices(self) -> tuple[int, ...]:
        return tuple(b.index for b in self.bins)

    def span(self, indices: list[int] | tuple[int, ...]) -> tuple[float, float]:
        """The seconds covered by a set of bin indices.

        Reported as one interval from the earliest start to the latest stop.
        A caller that hands in a gapped set gets the hull, which is the honest
        answer for an event that paused and resumed - the bin list stays in the
        record as the exact version.
        """
        chosen = [self.bins[i] for i in sorted(set(indices))]
        if not chosen:
            raise ValueError("no bins given")
        return (chosen[0].start, chosen[-1].stop)

    def covering(self, start: float, stop: float) -> tuple[int, ...]:
        """Bin indices that overlap the half-open interval [start, stop)."""
        if stop <= start:
            raise ValueError(f"empty interval [{start}, {stop})")
        return tuple(b.index for b in self.bins if b.start < stop and start < b.stop)

    def as_dict(self) -> dict:
        return {
            "fps": self.fps,
            "frames": self.frames,
            "duration": self.duration,
            "bin_seconds": self.bin_seconds,
            "count": self.count,
            "bins": [b.as_dict() for b in self.bins],
        }


def plan(
    frames: int,
    fps: float,
    *,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
    min_tail_seconds: float = DEFAULT_MIN_TAIL_SECONDS,
) -> Grid:
    """Lay a `bin_seconds` grid over `frames` frames at `fps`.

    Frame boundaries come from rounding the second boundaries once, so the
    bins partition the frames exactly: no frame belongs to two bins and none
    belongs to none. Deriving them any other way (a frames-per-bin constant,
    say) leaves a remainder at every non-integer frame rate.
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if bin_seconds <= 0:
        raise ValueError(f"bin_seconds must be positive, got {bin_seconds}")
    if min_tail_seconds < 0 or min_tail_seconds >= bin_seconds:
        raise ValueError(
            f"min_tail_seconds must be in [0, {bin_seconds}), got {min_tail_seconds}"
        )

    duration = frames / fps
    full = int(duration / bin_seconds + 1e-9)
    edges = [i * bin_seconds for i in range(full + 1)]
    tail = duration - edges[-1]
    if full == 0 or tail >= min_tail_seconds - 1e-9:
        # A clip shorter than one bin still gets a bin; it is just a short one.
        if tail > 1e-9 or full == 0:
            edges.append(duration)
    else:
        edges[-1] = duration

    bins: list[Bin] = []
    for index in range(len(edges) - 1):
        start, stop = edges[index], edges[index + 1]
        first_frame = round(start * fps)
        stop_frame = frames if index == len(edges) - 2 else round(stop * fps)
        bins.append(
            Bin(
                index=index,
                start=_round(start),
                stop=_round(stop),
                first_frame=first_frame,
                stop_frame=stop_frame,
            )
        )
    return Grid(fps=float(fps), frames=int(frames), bin_seconds=float(bin_seconds), bins=tuple(bins))


def from_dict(data: dict) -> Grid:
    """Rebuild a grid from what `Grid.as_dict` wrote."""
    bins = tuple(
        Bin(
            index=int(entry["index"]),
            start=_round(entry["t"][0]),
            stop=_round(entry["t"][1]),
            first_frame=int(entry["frames"][0]),
            stop_frame=int(entry["frames"][1]),
        )
        for entry in data["bins"]
    )
    return Grid(
        fps=float(data["fps"]),
        frames=int(data["frames"]),
        bin_seconds=float(data["bin_seconds"]),
        bins=bins,
    )


@dataclass(frozen=True)
class Cut:
    """How a window maps onto a grid: which bins survive, and re-based times."""

    kept: tuple[int, ...]
    offset: float
    grid: Grid

    def rebase(self, index: int) -> int:
        """Old bin index to new. Raises if that bin is not in the window."""
        return self.kept.index(index)


def cut(grid: Grid, start: float, stop: float) -> Cut:
    """Restrict a grid to [start, stop), producing a grid based at zero.

    Only bins fully inside the window are kept. A partially covered bin is
    dropped rather than truncated, because its caption describes a second the
    window does not entirely show, and a caption that overstates its window is
    worse for training than one second less of supervision. Windows cut on bin
    boundaries - which is every window a 1 s grid and integer-second cuts
    produce - lose nothing to this rule.
    """
    kept = tuple(b.index for b in grid.bins if b.start >= start - 1e-9 and b.stop <= stop + 1e-9)
    if not kept:
        raise ValueError(f"no whole bin lies inside [{start}, {stop})")
    chosen = [grid.bins[i] for i in kept]
    offset = chosen[0].start
    first_frame = chosen[0].first_frame
    bins = tuple(
        Bin(
            index=new_index,
            start=_round(b.start - offset),
            stop=_round(b.stop - offset),
            first_frame=b.first_frame - first_frame,
            stop_frame=b.stop_frame - first_frame,
        )
        for new_index, b in enumerate(chosen)
    )
    inner = Grid(
        fps=grid.fps,
        frames=bins[-1].stop_frame,
        bin_seconds=grid.bin_seconds,
        bins=bins,
    )
    return Cut(kept=kept, offset=_round(offset), grid=inner)
