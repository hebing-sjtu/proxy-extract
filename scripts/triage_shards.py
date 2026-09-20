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
# The note it prints after a traceback it caught and carried on from. Its
# presence after the last traceback is what distinguishes a shard that survived
# its errors from one that stopped at them.
HANDLED_TRAILER = "that is a traceback rather than a message about your data"
# The `[index/count]` the run prints before it starts. Anchored to the end of
# the line so it cannot match the `[3/812]` that heads every episode line.
BANNER = re.compile(r"\[(?P<index>\d+)/(?P<count>\d+)\]\s*$", re.MULTILINE)


def declared_shards(text: str) -> int | None:
    """How many shards the run that wrote this log split itself into.

    The launcher truncates the log of every shard it starts, but only of those
    it starts. Re-running with a smaller WORKERS_PER_GPU therefore leaves the
    surplus logs of the previous, wider run untouched on disk, where they read
    as part of this one - with their old failures, their old counts, and an
    ending that was never revisited. Asking each log which run wrote it is the
    only way to tell, because the mtimes of a run lasting hours overlap.
    """
    match = BANNER.search(text)
    return int(match.group("count")) if match else None


def classify(text: str) -> tuple[str, str]:
    """How this shard ended, as (verdict, detail)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "empty", "the log is empty - the worker never started"
    if FINISHED.search(lines[-1]):
        return "finished", lines[-1]

    # Whether it raised is decided by the *last* traceback, not by whether one
    # appears anywhere. Under --keep-going a healthy shard prints a traceback
    # for every episode it survives, so "contains a traceback" is true of
    # almost every log here and says nothing about how this one stopped.
    #
    # The two are told apart by the note `report_item_failure` prints after a
    # traceback it handled. A traceback with that trailer was survived; one
    # without it is where the shard stopped.
    _, marker, ending = text.rpartition("Traceback (most recent call last)")
    if not marker or HANDLED_TRAILER in ending:
        return "killed", f"stops after: {lines[-1][:120]}"

    if "No space left on device" in ending:
        return "disk full", "the output filesystem filled mid-run"
    if "CUDA out of memory" in ending or "OutOfMemoryError" in ending:
        return "cuda out of memory", _final_exception(ending)
    if "MemoryError" in ending or "Cannot allocate memory" in ending:
        return "host out of memory", _final_exception(ending)
    return "raised", _final_exception(ending)


def _final_exception(tail: str) -> str:
    """The exception a log ends on, rather than the last few lines of progress."""
    for line in reversed(tail.splitlines()):
        if re.match(r"^\S*(?:Error|Exception)\b", line.strip()):
            return line.strip()[:200]
    return tail.splitlines()[-1][:200] if tail.strip() else "(nothing)"


def error_kinds(text: str) -> dict[str, tuple[int, str]]:
    """Per-item failures inside a shard, as {kind: (count, one example)}.

    With an example, because the type alone does not say what to do: 43
    FileNotFoundError is a missing input if it names a source file and a
    self-inflicted wound if it names something under `.work`.
    """
    kinds: dict[str, tuple[int, str]] = {}

    def add(kind: str, example: str) -> None:
        count, first = kinds.get(kind, (0, example))
        kinds[kind] = (count + 1, first)

    for line in text.splitlines():
        typed = TYPED_ERROR.match(line)
        if typed:
            add(typed.group("kind"), line.split(": ", 2)[-1].strip())
            continue
        handled = ERROR_LINE.match(line)
        if handled:
            add("(handled: a message, not a bug)", handled.group("rest").strip())
    return kinds


Log = tuple[Path, str]


def shard_no(path: Path) -> int:
    return int(re.sub(r"\D", "", path.stem) or 0)


def brace(numbers: list[int]) -> str:
    """`[64, 65, ..., 127]` as `64..127`, so the printed `rm` can be pasted.

    Brace expansion takes no spaces, and a comma list of sixty-four shards is
    not something anyone should have to check by eye before running it.
    """
    runs: list[tuple[int, int]] = []
    for number in sorted(numbers):
        if runs and number == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], number)
        else:
            runs.append((number, number))
    return ",".join(f"{lo}..{hi}" if hi > lo else str(lo) for lo, hi in runs)


def this_run(read: list[Log]) -> tuple[list[Log], list[Log]]:
    """Split the logs into the current run's and an earlier run's leftovers.

    Logs that name different shard counts cannot be from one run. The current
    one is the group whose size its own logs agree with: a 64-shard run writes
    64 logs saying `/64`, while the 64 logs of the 128-shard run it replaced
    are the half that was not started again, and there are fewer of them than
    they claim. Falling back to mtime when that is ambiguous, which is the
    weaker signal - a long run's logs are written over hours.
    """
    by_count: dict[int | None, list[Log]] = defaultdict(list)
    for entry in read:
        by_count[declared_shards(entry[1])].append(entry)
    if len(by_count) < 2:
        return read, []

    def freshness(count: int | None) -> tuple[bool, float]:
        group = by_count[count]
        return (
            count is not None and len(group) == count,
            max(path.stat().st_mtime for path, _ in group),
        )

    current = max(by_count, key=freshness)
    stale = [entry for count, group in by_count.items() if count != current for entry in group]
    return by_count[current], sorted(stale, key=lambda entry: shard_no(entry[0]))


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

    read = [(path, path.read_text(errors="replace")) for path in logs]
    read, stale = this_run(read)
    if stale:
        spans = brace([shard_no(path) for path, _ in stale])
        print(
            f"ignoring {len(stale)} log(s) left by an earlier, wider run: shards {spans}.\n"
            "The launcher only truncates the logs of the shards it starts, so these\n"
            "still describe the previous run and would be counted as part of this one.\n"
            f"  rm {logs[0].parent}/shard-{{{spans}}}.log\n"
        )

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    kinds_total: dict[str, tuple[int, str]] = {}
    for path, text in read:
        verdict, detail = classify(text)
        grouped[verdict].append((path.stem.replace("shard-", ""), detail))
        for kind, (count, example) in error_kinds(text).items():
            running, first = kinds_total.get(kind, (0, example))
            kinds_total[kind] = (running + count, first)

    print(f"{len(read)} shard logs\n")
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
        print("per-episode failures inside the shards (survived, but the clips are missing):")
        for kind, (count, example) in sorted(kinds_total.items(), key=lambda item: -item[1][0]):
            print(f"  {count:6}  {kind}")
            print(f"          e.g. {example[:160]}")
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
