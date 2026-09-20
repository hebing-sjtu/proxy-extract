#!/usr/bin/env python3
"""Pick a review sample of clips, stratified by how much they flicker.

A hundred clips drawn at random answer almost nothing. Most of a driving corpus
is a road ahead with little in it, those clips look steady whatever the backend
did, and a reviewer who watches a hundred of them learns only that the median
case is fine - which was not in doubt. The cases that decide whether moge3 and
sam2 actually worked are the ones with enough happening to flicker, and random
sampling meets them at their corpus rate, which is low.

So this measures first and samples second, on the two axes the change was made
for:

  semantic flicker - the fraction of pixels whose class id differs from the
  previous frame. This is what sam2's masklets exist to hold still, and it is
  computed from `duv/*.semantic_id.png`, the delivered ids, not from anything
  upstream of them.

  depth jitter - the frame-to-frame step in the median log depth. A moving
  camera changes this smoothly; the metric head breathing changes it in steps,
  and `temporal.lock_depth_scale` exists to flatten exactly that. Measured as
  the median absolute step, so a few real cuts do not dominate a long clip.

The sample is then the worst clips on each axis plus an even spread across the
rest, because a reviewer needs to see both "is it fixed" and "where is it still
broken", and only the second one is hard to find.

    python scripts/select_clips.py CLIPS_DIR --count 100

Each pick is rendered as one mp4 with colour, depth and semantics side by side,
named so the worst cases sort to the top of the folder, and the whole directory
is a few hundred MB rather than the 5 GB the clips themselves come to.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy-extract" / "src"))

from proxy_extract import contract, preview, proxy_duv

# Depth is read in full - 124 frames of 258 KB - while semantics is a few KB a
# frame, so a pool of any size is dominated by depth I/O. Threads rather than
# processes because that wait is the filesystem, not the CPU.
READERS = 16

# How much of the sample is spent on the worst cases rather than on the spread.
# A quarter: enough that a real regression cannot be missed by bad luck, few
# enough that the reviewer still sees what the corpus mostly looks like.
WORST_SHARE = 0.25


@dataclass
class Scored:
    clip: str
    scene: str
    semantic_flicker: float
    depth_jitter: float
    picked_as: str = ""
    preview: str = ""


def complete_clips(root: Path) -> list[Path]:
    """Clips with everything a reviewer needs, skipping ones still being cut."""
    found = []
    for clip in sorted(root.glob("clip_*")):
        duv = proxy_duv.duv_dir_for(clip)
        if not (clip / "clip_report.json").is_file() or not duv.is_dir():
            continue
        if len(list(duv.glob("*.semantic_id.png"))) == proxy_duv.SPEC_FRAMES:
            found.append(clip)
    return found


def score(clip: Path) -> Scored | None:
    """Measure one clip on both flicker axes, or None if it cannot be read."""
    duv = proxy_duv.duv_dir_for(clip)
    try:
        report = json.loads((clip / "clip_report.json").read_text())
        labels, medians = [], []
        for ordinal in range(proxy_duv.SPEC_FRAMES):
            depth, semantic = contract.read_frame(duv, ordinal)
            labels.append(semantic)
            # Zero is "no depth" rather than a near reading, so it has to come
            # out before the median or a sky-heavy frame reads as close.
            valid = depth[depth > contract.DEPTH_VALID_EPSILON_METRES]
            medians.append(np.median(valid) if valid.size else np.nan)
    except (OSError, ValueError, KeyError):
        return None

    stack = np.stack(labels)
    changed = float(np.mean(stack[1:] != stack[:-1]))

    logs = np.log(np.asarray(medians, dtype=np.float64))
    steps = np.abs(np.diff(logs))
    steps = steps[np.isfinite(steps)]
    jitter = float(np.median(steps)) if steps.size else float("nan")

    return Scored(clip.name, report.get("scene", "?"), round(changed, 6), round(jitter, 6))


def one_per_episode(clips: list[Path], rng: random.Random) -> list[Path]:
    """At most one window per episode.

    Five windows of one episode are five views of the same street under the
    same models, so they cost five slots and answer barely more than one.
    """
    by_episode: dict[str, list[Path]] = defaultdict(list)
    for clip in clips:
        # `clip_000576_3` is window 3 of episode 576.
        by_episode[clip.name.rsplit("_", 1)[0]].append(clip)
    return [rng.choice(group) for group in by_episode.values()]


def stratify(scored: list[Scored], count: int) -> list[Scored]:
    """The worst on each axis, then an even spread over what is left."""
    worst_each = max(1, int(count * WORST_SHARE) // 2)
    picked: dict[str, Scored] = {}

    for axis in ("semantic_flicker", "depth_jitter"):
        ranked = sorted(
            (item for item in scored if np.isfinite(getattr(item, axis))),
            key=lambda item: -getattr(item, axis),
        )
        for item in ranked[:worst_each]:
            picked.setdefault(item.clip, item).picked_as = f"worst {axis}"

    # The spread is taken over the semantic axis because it is the one with a
    # meaningful zero: a clip with no label changes at all is genuinely static,
    # while a depth jitter of zero only says the camera held still.
    rest = sorted(
        (item for item in scored if item.clip not in picked),
        key=lambda item: item.semantic_flicker,
    )
    room = count - len(picked)
    if room > 0 and rest:
        for index in np.linspace(0, len(rest) - 1, num=min(room, len(rest))):
            item = rest[round(index)]
            picked.setdefault(item.clip, item).picked_as = "spread"

    return sorted(picked.values(), key=lambda item: item.clip)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips_dir", type=Path)
    parser.add_argument("--count", type=int, default=100, help="clips to select")
    parser.add_argument(
        "--pool",
        type=int,
        default=500,
        help="how many candidates to measure; 0 measures every clip",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path, help="where to write the mp4s (default CLIPS_DIR/review)")
    parser.add_argument("--fps", type=float, default=24.0, help="play rate; the clip's own is 24")
    args = parser.parse_args()

    root = args.clips_dir
    rng = random.Random(args.seed)
    clips = complete_clips(root)
    if not clips:
        print(f"no complete clips under {root}", file=sys.stderr)
        return 2

    candidates = one_per_episode(clips, rng)
    rng.shuffle(candidates)
    if args.pool:
        candidates = candidates[: args.pool]
    print(f"{len(clips)} complete clips, measuring {len(candidates)} of them")

    with ThreadPoolExecutor(max_workers=READERS) as pool:
        scored = [item for item in pool.map(score, candidates) if item is not None]
    unreadable = len(candidates) - len(scored)
    if unreadable:
        print(f"{unreadable} candidate(s) could not be read and were skipped")

    selection = stratify(scored, args.count)
    out_dir = args.out or root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Named so the worst cases sort to the top of a Drive folder listing. A
    # reviewer opening the folder should not have to consult a JSON file to
    # find out which of a hundred files is the one worth watching.
    rendered, failed = [], []
    for rank, item in enumerate(sorted(selection, key=lambda s: -s.semantic_flicker), start=1):
        name = f"{rank:03d}_{item.clip}_sem{item.semantic_flicker:.3f}.mp4"
        try:
            preview.render_clip_review(root / item.clip, out_dir / name, fps=args.fps)
        except (OSError, ValueError, RuntimeError) as error:
            failed.append(f"{item.clip}: {error}")
            continue
        item.preview = name
        rendered.append(item)
        print(f"  [{len(rendered)}/{len(selection)}] {name}")

    (out_dir / "selection.json").write_text(
        json.dumps(
            {
                "count": len(rendered),
                "measured": len(scored),
                "seed": args.seed,
                "clips": [asdict(item) for item in rendered],
            },
            indent=2,
        )
    )
    for note in failed:
        print(f"could not render {note}", file=sys.stderr)

    flicker = [item.semantic_flicker for item in rendered]
    print(
        f"\nrendered {len(rendered)} review clips "
        f"({sum(1 for item in rendered if item.picked_as.startswith('worst'))} worst, "
        f"{sum(1 for item in rendered if item.picked_as == 'spread')} spread)\n"
        f"semantic flicker across the sample: {min(flicker):.4f} to {max(flicker):.4f}\n"
        f"  {out_dir}\n\n"
        f"upload with:\n"
        f"  rclone copy {out_dir} gdrive:clips-review --progress --transfers 8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
