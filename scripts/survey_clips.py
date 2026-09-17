#!/usr/bin/env python3
"""What made the clips sitting in an output root, grouped.

An output directory says how many clips it holds and nothing about where they
came from, which is the question that matters the moment a corpus is re-cut
with different models. A root that looks 99% finished may be 99% finished with
the backend you were trying to replace.

    python scripts/survey_clips.py /data/binghe/datasets/ABot-sub-2000-clips

Reads each clip_report.json and counts the combinations of depth backend,
semantic refiner and whether PROXY_DUV_SPEC's per-frame form is present. Also
reports clips whose depth videos predate the full-range encode fix, since those
cannot be told apart by looking at the pixels.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} CLIPS_DIR", file=sys.stderr)
        print("  groups the clips in an output root by what produced them", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    reports = sorted(root.glob("*/clip_report.json"))
    if not reports:
        print(f"{root} holds no clips (no */clip_report.json)")
        return 0

    combinations: Counter[tuple] = Counter()
    unreadable = 0
    oldest = newest = None
    for path in reports:
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable += 1
            continue
        combinations[
            (
                (report.get("depth") or {}).get("backend"),
                (report.get("semantic") or {}).get("refiner"),
                bool(report.get("proxy_duv")),
            )
        ] += 1
        stamp = path.stat().st_mtime
        oldest = stamp if oldest is None else min(oldest, stamp)
        newest = stamp if newest is None else max(newest, stamp)

    print(f"{root}")
    print(f"  {len(reports)} clips\n")
    print(f"  {'depth backend':22} {'refiner':10} {'proxy_duv':10} count")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10} -----")
    for (depth, refiner, duv), count in combinations.most_common():
        print(f"  {depth or '?':22} {refiner or 'none':10} {duv!s:10} {count}")
    if unreadable:
        print(f"  {'(unreadable report)':22} {'':10} {'':10} {unreadable}")

    if oldest is not None:
        import datetime as dt

        # Local time, because the question this answers is "was this before or
        # after I ran that", and the operator's clock is the one being compared.
        def when(stamp: float) -> str:
            moment = dt.datetime.fromtimestamp(stamp, tz=dt.UTC).astimezone()
            return moment.strftime("%Y-%m-%d %H:%M")

        print(f"\n  written between {when(oldest)} and {when(newest)}")

    if len(combinations) > 1:
        print(
            "\nMore than one combination is present, so this root is a mixture. Nothing\n"
            "downstream distinguishes them - a training run would see one corpus. If the\n"
            "intent was to replace the old predictions, cut into a fresh CLIPS_DIR."
        )
    print(
        "\nNote on depth: a clip cut on a node whose ffmpeg was Ubuntu's 4.4.2 carries an\n"
        "untagged depth plane, which reads back rescaled. Neither this survey nor the\n"
        "pixels can show it, because the codes are intact and only the range tag is\n"
        "missing. Timestamps are the only evidence: clips written before this node's\n"
        "depth encode was fixed should be treated as suspect rather than inspected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
