#!/usr/bin/env python3
"""Pre-download model weights into the mounted HF cache.

Run this once per machine, with network. Afterwards every extraction can run
with `HF_HUB_OFFLINE=1`, which is not just tidiness: a batch job that hits the
hub per worker will rate-limit itself, and a hub outage halfway through a
sharded run leaves a partially processed dataset.

The repo names are imported from the backends rather than restated here, so a
checkpoint swap cannot leave this script fetching the wrong thing.
"""

from __future__ import annotations

import argparse
import sys

from proxy_extract.depth.depth_anything import INDOOR_CHECKPOINT, OUTDOOR_CHECKPOINT
from proxy_extract.depth.depth_anything_v3 import METRIC_CHECKPOINT, NESTED_CHECKPOINT
from proxy_extract.depth.mapanything import APACHE_CHECKPOINT, DEFAULT_CHECKPOINT
from proxy_extract.semantic.panoptic import ADE20K_CHECKPOINT, CITYSCAPES_CHECKPOINT

# Grouped by what a run actually needs, so nobody waits on a 6 GB download for
# a backend they are not using.
SETS: dict[str, tuple[str, ...]] = {
    # The recommended configuration: --semantic-backend coarse6 --depth-backend depth_anything
    "default": (ADE20K_CHECKPOINT, OUTDOOR_CHECKPOINT),
    "semantic": (ADE20K_CHECKPOINT, CITYSCAPES_CHECKPOINT),
    "depth": (OUTDOOR_CHECKPOINT, INDOOR_CHECKPOINT),
    # Gated on the hub and CC-BY-NC. Fetching needs `huggingface-cli login`
    # after accepting the terms; the -apache mirror is the one to ship with.
    "mapanything": (DEFAULT_CHECKPOINT, APACHE_CHECKPOINT),
    # 6.8 GB, and CC-BY-NC, so it is deliberately not in `default`. Unlike
    # mapanything it needs nothing from outside the hub: the DINOv2 backbone
    # ships inside the checkpoint rather than through torch.hub.
    "da3": (NESTED_CHECKPOINT,),
    "da3-apache": (METRIC_CHECKPOINT,),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", dest="sets", nargs="+", default=["default"], choices=sorted(SETS),
        help="which groups of weights to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="report and continue when a repo is gated or unreachable",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    wanted = sorted({repo for name in args.sets for repo in SETS[name]})
    failed: list[tuple[str, str]] = []

    for repo in wanted:
        print(f"--> {repo}", flush=True)
        try:
            path = snapshot_download(repo_id=repo)
        except Exception as error:  # gated repo, no token, no network
            print(f"    FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            failed.append((repo, f"{type(error).__name__}: {error}"))
            if not args.keep_going:
                return 1
            continue
        print(f"    ok  {path}", flush=True)

    if failed:
        print(f"\n{len(failed)}/{len(wanted)} failed:", file=sys.stderr)
        for repo, reason in failed:
            print(f"  {repo}: {reason}", file=sys.stderr)
        return 1

    print(f"\n{len(wanted)} repo(s) cached. Runs can now set HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
