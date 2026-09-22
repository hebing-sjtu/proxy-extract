#!/usr/bin/env python3
"""Backfill `duv/semantic.json` into an existing PROXY_DUV corpus.

New clips get this sidecar while their first frame is written. This command is
for a corpus produced before the sidecar existed:

    python scripts/write_semantic_uv.py /data/.../ABot-sub-2000-clips-moge3

The same canonical mapping is written into every DUV directory, as required by
PROXY_DUV_SPEC.md. Files are tiny; duplicating the mapping makes each clip
self-describing when copied out of the corpus on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy-extract" / "src"))

from proxy_extract import proxy_duv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="clips or segments root")
    parser.add_argument("--check", action="store_true", help="report missing files without writing")
    args = parser.parse_args()

    duv_dirs = sorted(path for path in args.root.glob("*/duv") if path.is_dir())
    if not duv_dirs:
        print(f"no */duv directories under {args.root}", file=sys.stderr)
        return 2

    missing = [path for path in duv_dirs if not (path / proxy_duv.SEMANTIC_NAME).is_file()]
    if args.check:
        print(f"{len(duv_dirs)} DUV directories; {len(missing)} missing semantic.json")
        return 1 if missing else 0

    for path in duv_dirs:
        # Rewrite all, not just missing files: an earlier hand-written mapping
        # is more dangerous than no mapping because it looks authoritative.
        proxy_duv.write_semantic_json(path)
    print(f"{len(duv_dirs)} DUV directories -> semantic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
