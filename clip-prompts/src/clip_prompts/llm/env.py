"""Reading credentials out of the environment and out of `.env` files."""

from __future__ import annotations

import os
from pathlib import Path


def env_any(*names: str, default: str = "") -> str:
    """First non-empty value among `names`.

    Several of these settings have accumulated more than one spelling over the
    life of the sibling pipeline, and the credentials people already have on
    disk use the old ones. Accepting aliases is cheaper than asking everyone to
    rename their keys.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def load_dotenv(path: Path) -> None:
    """Load `KEY=VALUE` pairs from a file, never overriding what is already set.

    Not overriding is the important half: it means a node can export one
    variable to point at a different project without editing a file that is
    shared with everything else running there.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().rstrip(",").strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def load_env(*roots: Path) -> list[Path]:
    """Load `.env` then `.env.local` from each root, and report what was read.

    The list comes back so a caller can print it. A run that silently picked up
    no credentials and a run that picked up the wrong ones look identical from
    the outside until something 401s several minutes in.
    """
    read: list[Path] = []
    for root in roots:
        for name in (".env", ".env.local"):
            path = root / name
            if path.is_file():
                load_dotenv(path)
                read.append(path)
    return read
