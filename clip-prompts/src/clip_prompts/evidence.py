"""What each second of a clip contains, read off the DUV instead of guessed.

Every clip already ships a proxy stream whose red channel is a log-depth code
and whose green and blue are a class palette. Decoding it costs no API call and
no GPU, and it answers exactly the questions a captioner is worst at and most
confidently wrong about: how many people are on screen, whether there is a
vehicle, whether the protagonist is visible at all, how far away the world is,
and whether anything moved.

That makes this module two things at once.

**A prior.** The per-second table goes into the captioner's prompt. Told that
second 3 contains one protagonist and two other people and no vehicle, a model
stops inventing the parked car it expects to see on a street.

**A referee.** The same table is what `verify` holds the reply against. A
caption that describes a car in a second with no vehicle pixels is wrong in a
way that is detectable for free, at scale, on all 9,985 clips - and a corpus
you can filter is worth more than one you have to trust.

Two limits of the proxy are load-bearing, both from `DATA_CLIPS.md`. The class
palette is not injective: building, ground, terrain, water and prop all encode
as (0, 0), so this module reports "static other" and never those five. And the
protagonist / bystander split comes from a tracker that can fail, in which case
everyone lands in `ped`; `hero_resolved=False` in the report is the signal to
stop treating a zero player count as evidence of anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .timeline import Grid

# The DUV depth code, from DATA_CLIPS.md section 3. 255 is a sentinel for sky
# *or* failed depth, so it is missing data rather than "8000 m away".
NEAR_METRES = 0.1
FAR_METRES = 8000.0
MAX_CODE = 254.0
SKY_CODE = 255

# Class groups the palette can actually be inverted to. Eight, not eleven.
GROUP_STATIC = 0
GROUP_ROAD = 1
GROUP_SKY = 2
GROUP_PLAYER = 3
GROUP_PED = 4
GROUP_VEHICLE = 5
GROUP_EGO = 6
GROUP_VEGETATION = 7

GROUP_NAMES = {
    GROUP_STATIC: "static",
    GROUP_ROAD: "road",
    GROUP_SKY: "sky",
    GROUP_PLAYER: "player",
    GROUP_PED: "ped",
    GROUP_VEHICLE: "vehicle",
    GROUP_EGO: "ego_vehicle",
    GROUP_VEGETATION: "vegetation",
}

# A 336x192 frame is 64,512 pixels. Eight is about a person at 60 m; below it
# the blob is as likely to be a mislabelled railing as a bystander.
PED_MIN_PIXELS = 8

# Mean absolute change in the log-depth code between neighbouring frames, over
# pixels with valid depth. One code is 4.4% of a distance, so this is "the
# world came a couple of percent closer or further, on average, in a frame".
#
# This is the one number in this package that has not been fitted on this
# corpus, and it only drives a warning, never a rejection. The way to fit it is
# to run `captions-audit` on a few hundred clips and look at how often the
# nothing-moves warning fires: a threshold that is too low fires on every
# handheld shot, and one that is too high never fires at all.
STATIC_CODE_DELTA = 0.5


def depth_metres(red: np.ndarray) -> np.ndarray:
    """DUV red channel to metres. Sky and invalid depth come back as NaN."""
    code = np.asarray(red, dtype=np.float64)
    metres = NEAR_METRES * (FAR_METRES / NEAR_METRES) ** (code / MAX_CODE)
    return np.where(code >= SKY_CODE, np.nan, metres)


def code_to_metres(code: float) -> float:
    return NEAR_METRES * (FAR_METRES / NEAR_METRES) ** (code / MAX_CODE)


def classes(frame: np.ndarray) -> np.ndarray:
    """One DUV RGB frame to the eight invertible groups."""
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    out = np.zeros(r.shape, np.uint8)
    gb = (g.astype(np.uint16) << 8) | b
    white = gb == ((255 << 8) | 255)
    out[white] = GROUP_ROAD
    out[white & (r == SKY_CODE)] = GROUP_SKY
    out[gb == 255] = GROUP_PLAYER
    out[gb == 128] = GROUP_PED
    out[gb == (64 << 8)] = GROUP_VEHICLE
    out[gb == (128 << 8)] = GROUP_EGO
    out[gb == (255 << 8)] = GROUP_VEGETATION
    return out


def _frames(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            yield bgr[:, :, ::-1]
    finally:
        capture.release()


def _blobs(mask: np.ndarray, *, min_pixels: int) -> int:
    import cv2

    if not mask.any():
        return 0
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0
    sizes = np.bincount(labels.ravel(), minlength=count)[1:]
    return int((sizes >= min_pixels).sum())


def _bbox(mask: np.ndarray) -> list[float] | None:
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    height, width = mask.shape
    return [
        round(float(cols[0]) / width, 3),
        round(float(rows[0]) / height, 3),
        round(float(cols[-1] + 1) / width, 3),
        round(float(rows[-1] + 1) / height, 3),
    ]


@dataclass
class _Accumulator:
    """Per-bin running totals, so a long episode never holds all its frames."""

    frames: int = 0
    fractions: np.ndarray = None  # type: ignore[assignment]
    histogram: np.ndarray = None  # type: ignore[assignment]
    ped_counts: list = None  # type: ignore[assignment]
    player_boxes: list = None  # type: ignore[assignment]
    deltas: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.fractions = np.zeros(len(GROUP_NAMES), dtype=np.float64)
        self.histogram = np.zeros(256, dtype=np.int64)
        self.ped_counts = []
        self.player_boxes = []
        self.deltas = []


def _percentile_code(histogram: np.ndarray, quantile: float) -> float | None:
    """Percentile of the valid depth codes, straight out of the histogram.

    Exact rather than sampled, and constant memory: 255 possible codes means
    the histogram *is* the distribution, so there is no reason to keep frames
    around to compute this from.
    """
    valid = histogram[:SKY_CODE]
    total = int(valid.sum())
    if total == 0:
        return None
    target = quantile * total
    running = 0
    for code, count in enumerate(valid):
        running += int(count)
        if running >= target:
            return float(code)
    return float(SKY_CODE - 1)


def measure(duv: Path, grid: Grid, *, hero_resolved: bool = True) -> dict:
    """Per-second facts for one clip's DUV file."""
    duv = Path(duv)
    if not duv.is_file():
        raise FileNotFoundError(duv)
    return measure_frames(_frames(duv), grid, hero_resolved=hero_resolved, source=str(duv))


def measure_frames(frames, grid: Grid, *, hero_resolved: bool = True, source: str = "duv") -> dict:
    """Per-second facts from a stream of DUV RGB frames.

    Takes an iterable rather than a path so that a caller with the frames
    already in hand - the clip cutter, a test, a future episode-level pass -
    does not have to encode them to a file first. Frames are folded into their
    bin as they arrive, so this costs one frame of memory whether the input is
    five seconds or a minute.
    """
    owner = np.empty(grid.frames, dtype=np.int64)
    for item in grid.bins:
        owner[item.first_frame : item.stop_frame] = item.index

    accumulators = [_Accumulator() for _ in grid.bins]
    previous: np.ndarray | None = None
    seen = 0

    for index, frame in enumerate(frames):
        if index >= grid.frames:
            break
        seen += 1
        acc = accumulators[int(owner[index])]
        acc.frames += 1

        red = frame[..., 0]
        group = classes(frame)
        counts = np.bincount(group.ravel(), minlength=len(GROUP_NAMES))
        acc.fractions += counts[: len(GROUP_NAMES)] / group.size
        acc.histogram += np.bincount(red.ravel(), minlength=256)

        acc.ped_counts.append(_blobs(group == GROUP_PED, min_pixels=PED_MIN_PIXELS))
        box = _bbox(group == GROUP_PLAYER)
        if box is not None:
            acc.player_boxes.append(box)

        if previous is not None:
            valid = (red < SKY_CODE) & (previous < SKY_CODE)
            if valid.any():
                delta = np.abs(red[valid].astype(np.int16) - previous[valid].astype(np.int16))
                acc.deltas.append(float(delta.mean()))
        previous = red

    if seen < grid.frames:
        raise ValueError(f"{source} decoded {seen} frames, expected {grid.frames}")

    rows = []
    for item, acc in zip(grid.bins, accumulators):
        share = acc.fractions / max(acc.frames, 1)
        boxes = np.asarray(acc.player_boxes) if acc.player_boxes else None
        motion = float(np.mean(acc.deltas)) if acc.deltas else 0.0
        row = {
            "bin": item.index,
            "t": [item.start, item.stop],
            "player": bool(share[GROUP_PLAYER] > 0),
            "peds": int(np.median(acc.ped_counts)) if acc.ped_counts else 0,
            "vehicle": bool(share[GROUP_VEHICLE] > 0),
            "ego_vehicle": bool(share[GROUP_EGO] > 0),
            "player_area": round(float(share[GROUP_PLAYER]), 5),
            "sky_frac": round(float(share[GROUP_SKY]), 3),
            "road_frac": round(float(share[GROUP_ROAD]), 3),
            "vegetation_frac": round(float(share[GROUP_VEGETATION]), 3),
            "static_frac": round(float(share[GROUP_STATIC]), 3),
            "player_bbox": (
                [round(float(v), 3) for v in np.median(boxes, axis=0)]
                if boxes is not None
                else None
            ),
            "depth_delta_code": round(motion, 3),
            "world_moving": bool(motion > STATIC_CODE_DELTA),
        }
        for name, quantile in (("depth_p10_m", 0.10), ("depth_p50_m", 0.50), ("depth_p90_m", 0.90)):
            code = _percentile_code(acc.histogram, quantile)
            row[name] = None if code is None else round(code_to_metres(code), 1)
        rows.append(row)

    return {
        "source": "duv",
        "from": source,
        "hero_resolved": bool(hero_resolved),
        "ped_min_pixels": PED_MIN_PIXELS,
        "static_code_delta": STATIC_CODE_DELTA,
        "per_bin": rows,
    }


def briefing(evidence: dict) -> str:
    """The per-second table as the few lines that go into the VLM prompt.

    Terse on purpose. This competes for attention with the instructions, and a
    model given a wall of floats starts transcribing them into the caption.
    """
    lines: list[str] = []
    for row in evidence.get("per_bin", []):
        parts = [f"bin {row['bin']:>2} ({row['t'][0]:g}-{row['t'][1]:g}s):"]
        parts.append("protagonist visible" if row["player"] else "protagonist not visible")
        peds = row["peds"]
        parts.append("no other people" if peds == 0 else f"{peds} other {'person' if peds == 1 else 'people'}")
        if row["ego_vehicle"]:
            parts.append("viewer is in a vehicle")
        parts.append("vehicle on screen" if row["vehicle"] else "no vehicle")
        if row["depth_p50_m"] is not None:
            parts.append(f"median depth {row['depth_p50_m']:g} m")
        parts.append(f"sky {round(row['sky_frac'] * 100)}%")
        parts.append("world moving" if row["world_moving"] else "world nearly still")
        lines.append(" ".join([parts[0], ", ".join(parts[1:])]))

    if not evidence.get("hero_resolved", True):
        lines.append(
            "NOTE: the protagonist tracker did not resolve on this clip, so every person "
            "is counted as a bystander. Treat 'protagonist not visible' as unknown here."
        )
    return "\n".join(lines)


def contact_sheet(rgb: Path, grid: Grid, out: Path, *, columns: int = 4, tile_width: int = 336) -> Path:
    """One labelled frame per bin, tiled, to anchor bin indices to pixels.

    A model handed a video and told "bin 3 is 3 to 4 seconds" has to trust its
    own sense of elapsed time. A sheet whose third tile is stamped `3` removes
    that step, which is the single cheapest defence against events landing a
    second early or late. Sent alongside the video, not instead of it - one
    frame a second cannot show gait.
    """
    import cv2

    rgb = Path(rgb)
    if not rgb.is_file():
        raise FileNotFoundError(rgb)

    wanted = {}
    for item in grid.bins:
        wanted[(item.first_frame + item.stop_frame) // 2] = item

    picked: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(rgb))
    try:
        index = 0
        while len(picked) < len(wanted):
            ok, bgr = capture.read()
            if not ok:
                break
            if index in wanted:
                picked[wanted[index].index] = bgr
            index += 1
    finally:
        capture.release()
    if len(picked) != len(wanted):
        raise ValueError(f"{rgb} gave {len(picked)} of {len(wanted)} sheet frames")

    height = max(1, int(tile_width * picked[0].shape[0] / picked[0].shape[1]))
    columns = max(1, min(columns, len(picked)))
    rows = math.ceil(len(picked) / columns)
    sheet = np.zeros((rows * height, columns * tile_width, 3), np.uint8)

    for position, item in enumerate(grid.bins):
        tile = cv2.resize(picked[item.index], (tile_width, height), interpolation=cv2.INTER_AREA)
        label = f"{item.index}  {item.start:g}-{item.stop:g}s"
        cv2.rectangle(tile, (0, 0), (tile_width, 26), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        top = (position // columns) * height
        left = (position % columns) * tile_width
        sheet[top : top + height, left : left + tile_width] = tile

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), sheet):
        raise RuntimeError(f"could not write {out}")
    return out
