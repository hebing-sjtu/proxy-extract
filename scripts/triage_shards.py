#!/usr/bin/env python3
"""Group the shard logs of a run by how they ended.

Seven shards failing prints seven paths and nothing about whether they failed
for one reason or seven, and the distinction that matters most is not in the
error text at all:

  a shard that **raised** ends with a traceback. It hit a bug or a bad input,
  and the text says which.

  a shard that was **killed** ends mid-sentence with nothing. Nothing was
  raised, so nothing was logged. That is the signature of the OOM killer, and
  it means the node was over-subscribed rather than the code being wrong -
  which is the opposite response: lower WORKERS_PER_GPU, do not debug.

    python scripts/triage_shards.py /data/binghe/datasets/ABot-sub-2000-clips-moge3

Reads only the logs, so it is safe to run while the shards are still going.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# What `clip-episodes` prints as its last line when it completes, whatever the
# per-episode outcome was. Its absence is the whole signal for a killed shard.
FINISHED = re.compile(r"\d+ clips, \d+ episodes failed, manifest at ")
ERROR_LINE = re.compile(r"^error: (?P<item>\S+?): (?P<rest>.*)$")
# `report_item_failure` puts the type in front for anything unexpected.
TYPED_ERROR = re.compile(r"^error: \S+?: (?P<kind>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)): ")
TAIL_LINES = 12


def classify(text: str) -> tuple[str, str]:
    """How this shard ended, as (verdict, detail)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "empty", "the log is empty - the worker never started"
    if FINISHED.search(lines[-1]):
        return "finished", lines[-1]

    # Nothing raised on the way out. Python prints a traceback for any
    # exception that reaches the top, and the launcher redirects stderr here,
    # so a log that simply stops was stopped from outside.
    tail = "\n".join(lines[-3:])
    if "Traceback (most recent call last)" not in text and "error:" not in tail:
        return "killed", f"stops after: {lines[-1][:120]}"
    if "MemoryError" in text or "Cannot allocate memory" in text:
        return "out of memory", "the allocation failed rather than being killed"
    if "CUDA out of memory" in text:
        return "cuda out of memory", "this one is per-worker; lower WORKERS_PER_GPU"
    if "No space left on device" in text:
        return "disk full", "the output filesystem filled mid-run"
    return "raised", "\n".join(lines[-3:])


def error_kinds(text: str) -> dict[str, int]:
    """Per-item failures inside a shard, counted by exception type."""
    kinds: dict[str, int] = defaultdict(int)
    for line in text.splitlines():
        typed = TYPED_ERROR.match(line)
        if typed:
            kinds[typed.group("kind")] += 1
            continue
        if ERROR_LINE.match(line):
            kinds["(handled: a message, not a bug)"] += 1
    return dict(kinds)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} CLIPS_DIR", file=sys.stderr)
        return 2
    logs = sorted(
        (Path(sys.argv[1]) / "logs").glob("shard-*.log"),
        key=lambda path: int(re.sub(r"\D", "", path.stem) or 0),
    )
    if not logs:
        print(f"no shard logs under {sys.argv[1]}/logs", file=sys.stderr)
        return 2

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    kinds_total: dict[str, int] = defaultdict(int)
    for path in logs:
        text = path.read_text(errors="replace")
        verdict, detail = classify(text)
        grouped[verdict].append((path.stem.replace("shard-", ""), detail))
        for kind, count in error_kinds(text).items():
            kinds_total[kind] += count

    print(f"{len(logs)} shard logs\n")
    for verdict in sorted(grouped, key=lambda name: -len(grouped[name])):
        entries = grouped[verdict]
        shards = ", ".join(shard for shard, _ in entries)
        print(f"{verdict}: {len(entries)}")
        print(f"  shards {shards}")
        if verdict != "finished":
            # One example is enough when they share a verdict; the point here
            # is the grouping, and the log itself is one command away.
            print(f"  e.g. {entries[0][1]}".replace("\n", "\n       "))
        print()

    if kinds_total:
        print("per-episode failures inside the shards, by kind:")
        for kind, count in sorted(kinds_total.items(), key=lambda item: -item[1]):
            print(f"  {count:6}  {kind}")
        print()

    if "killed" in grouped:
        print(
            f"{len(grouped['killed'])} shard(s) were killed rather than failing. Nothing was\n"
            "raised, so there is nothing in the logs to fix. On a node running many\n"
            "workers this is almost always the OOM killer, and the kernel records it:\n"
            "  dmesg -T | grep -i -E 'out of memory|killed process' | tail -20\n"
            "If it is, lower WORKERS_PER_GPU and re-run - the finished clips are kept."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
