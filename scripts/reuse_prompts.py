#!/usr/bin/env python3
"""Carry captions over from an earlier cut of the same corpus.

Captioning ten thousand clips is the one step in this pipeline that costs money
per clip, and re-cutting with different depth and semantic backends does not
change a word of what the caption says: the prose describes the picture, and
the picture is made of the same source frames either way.

What makes that safe is that the windowing is a pure function of the episode.
`plan_windows` divides an episode into equal buckets and centres a window in
each, with no seed, so the same episode cut with the same `--per-scene`,
`--frames` and `--fps` yields the same frames under the same name. What it is
*not* a function of is the depth backend, the refiner, or `--proxy-duv`.

But "the rule did not change" is not the same as "the two runs were given the
same arguments", and nothing about a caption looks wrong when it is describing
the wrong five seconds. So this verifies rather than assumes: a caption is
carried across only when both cuts record the identical `source_ordinals` for
that clip name, which is the list of source frames the clip is actually made
of.

Only `annotations/prompt.json` is copied. `prompt.txt` is deliberately left
behind - it is the export of a caption that passed verification against the
*old* DUV, and whether it still passes is a question for the new one:

    python scripts/reuse_prompts.py OLD_CLIPS NEW_CLIPS
    python -m clip_prompts captions-recompile --clips NEW_CLIPS --reverify
    python -m clip_prompts captions-export --clips NEW_CLIPS --write-txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPORT = "clip_report.json"
PROMPT = Path("annotations") / "prompt.json"


def ordinals_of(clip: Path) -> list[int] | None:
    """The source frames this clip is made of, or None if it cannot be read."""
    try:
        report = json.loads((clip / REPORT).read_text())
    except (OSError, ValueError):
        return None
    found = report.get("source_ordinals")
    return list(found) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path, help="the cut whose captions exist")
    parser.add_argument("new", type=Path, help="the cut that needs them")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace a caption already there")
    args = parser.parse_args()

    tally = {"copied": 0, "already": 0, "no caption": 0, "not in the old cut": 0, "different frames": 0}
    mismatched: list[str] = []

    for clip in sorted(args.new.glob("clip_*")):
        old = args.old / clip.name
        if not (old / REPORT).is_file():
            tally["not in the old cut"] += 1
            continue
        if not (old / PROMPT).is_file():
            tally["no caption"] += 1
            continue
        if (clip / PROMPT).is_file() and not args.overwrite:
            tally["already"] += 1
            continue

        # The whole safety argument reduces to this comparison.
        if ordinals_of(old) != ordinals_of(clip) or ordinals_of(clip) is None:
            tally["different frames"] += 1
            mismatched.append(clip.name)
            continue

        if not args.dry_run:
            (clip / PROMPT).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old / PROMPT, clip / PROMPT)
        tally["copied"] += 1

    width = max(len(key) for key in tally)
    print(f"{'(dry run) ' if args.dry_run else ''}captions:")
    for key, count in tally.items():
        print(f"  {count:6}  {key.ljust(width)}")

    if mismatched:
        print(
            f"\n{len(mismatched)} clip(s) cover different source frames in the two cuts,\n"
            "e.g. " + ", ".join(mismatched[:5]) + ".\n"
            "Their captions describe other footage and were not copied. That means the\n"
            "two runs were cut with different --per-scene, --frames or --fps, so check\n"
            "those before trusting the ones that did match.",
            file=sys.stderr,
        )
    if tally["copied"]:
        print(
            "\nthe copied captions were verified against the old DUV, so re-check them\n"
            "against the new one before exporting:\n"
            f"  python -m clip_prompts captions-recompile --clips {args.new} --reverify\n"
            f"  python -m clip_prompts captions-export --clips {args.new} --write-txt"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
