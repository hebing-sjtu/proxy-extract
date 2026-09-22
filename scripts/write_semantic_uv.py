#!/usr/bin/env python3
"""Write one corpus-wide `semantic.json` into an existing PROXY_DUV root.

Every clip uses the same class ids and UV bytes, so the mapping belongs beside
the clips rather than inside each one:

    python scripts/write_semantic_uv.py /data/.../ABot-sub-2000-clips-moge3

This also removes per-clip copies written by the short-lived first
implementation. One authoritative file is safer than ten thousand identical
files that can later disagree.
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
    parser.add_argument("--check", action="store_true", help="check the root file without writing")
    args = parser.parse_args()

    duv_dirs = sorted(path for path in args.root.glob("*/duv") if path.is_dir())
    if not duv_dirs:
        print(f"no */duv directories under {args.root}", file=sys.stderr)
        return 2

    path = args.root / proxy_duv.SEMANTIC_NAME
    duplicates = [duv / proxy_duv.SEMANTIC_NAME for duv in duv_dirs]
    duplicates = [duplicate for duplicate in duplicates if duplicate.is_file()]
    if args.check:
        print(
            f"{len(duv_dirs)} DUV directories; root semantic.json "
            f"{'present' if path.is_file() else 'missing'}; "
            f"{len(duplicates)} per-clip duplicate(s)"
        )
        return 0 if path.is_file() and not duplicates else 1

    proxy_duv.write_semantic_json(args.root)
    for duplicate in duplicates:
        duplicate.unlink()
    print(
        f"{len(duv_dirs)} DUV directories share {path}; "
        f"removed {len(duplicates)} per-clip duplicate(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
