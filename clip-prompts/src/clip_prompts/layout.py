"""Where a clip's parts are, and where its caption goes.

The corpus this reads is `DATA_CLIPS.md`'s: one directory per clip, holding
`target/rgb.mp4`, `proxy/duv.mp4`, an `annotations/` directory that may not
exist, and `clip_report.json`. The caption is written to
`<clip>/annotations/prompt.json` - inside the directory the corpus already
uses for its own claims about that clip, because a reader looking for what is
known about a clip looks there, and a parallel tree keyed by clip name is one
rename away from silently pairing captions with the wrong video.

`annotations/` is created when it is missing. Roughly a fifth of the clips
have no annotations at all, and refusing to caption those would leave holes in
the training set for a reason that has nothing to do with the clips.

There is no corpus-wide manifest, by design upstream: the 112 shard manifests
never got merged and the tools walk the tree instead. `discover` does the same,
so a caption run picks up clips that landed after it started.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contract import PROMPT_NAME

CLIP_GLOB = "clip_*"
REPORT_NAME = "clip_report.json"
ANNOTATIONS = "annotations"


@lru_cache(maxsize=4096)
def _read_report(path: Path) -> dict:
    """Reports are read four or five times per clip and never change under a run."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Clip:
    """One clip directory, resolved."""

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def rgb(self) -> Path:
        return self.root / "target" / "rgb.mp4"

    @property
    def anchor(self) -> Path:
        return self.root / "target" / "anchor.png"

    @property
    def duv(self) -> Path:
        return self.root / "proxy" / "duv.mp4"

    @property
    def report_path(self) -> Path:
        return self.root / REPORT_NAME

    @property
    def annotations(self) -> Path:
        return self.root / ANNOTATIONS

    @property
    def prompt(self) -> Path:
        return self.annotations / PROMPT_NAME

    @property
    def sheet(self) -> Path:
        """Working file, kept beside the caption so a bad bin can be seen."""
        return self.annotations / "prompt_sheet.jpg"

    @property
    def prompt_txt(self) -> Path:
        """The exported CWM user sentence.

        At the clip root rather than in `annotations/`, which is the one place
        in this package where that is right: FastVideo's manifest builder reads
        `<clip>/prompt.txt` and falls back to the episode caption if it is
        missing, so this path is fixed by a consumer rather than chosen here.
        """
        return self.root / "prompt.txt"

    def report(self) -> dict:
        return _read_report(self.report_path)

    def shape(self) -> tuple[int, float]:
        """(frames, fps) from the report, which is authoritative over the mp4."""
        report = self.report()
        return int(report["frames"]), float(report["fps"])

    def hero_resolved(self) -> bool:
        """Whether the protagonist tracker resolved on this clip.

        Two-step clips do not carry the field at all - `DATA_CLIPS.md` says it
        only appears in the one-pass reports - so a missing value is read as
        resolved. Reading it as unresolved would switch off the protagonist
        check for the entire subset, which is where the check is most useful.
        """
        semantic = self.report().get("semantic") or {}
        split = semantic.get("hero_split") or {}
        return bool(split.get("resolved", True))

    def usable(self) -> bool:
        """False for clips the extractor marked as placeholder-backend output."""
        return bool(self.report().get("deliverable", True))

    def check(self) -> None:
        for path in (self.rgb, self.duv, self.report_path):
            if not path.is_file():
                raise FileNotFoundError(path)


def clip_at(path: Path | str) -> Clip:
    return Clip(Path(path).expanduser().resolve())


def is_clip(path: Path) -> bool:
    return path.is_dir() and (path / REPORT_NAME).is_file()


def discover(root: Path | str, *, limit: int | None = None) -> list[Clip]:
    """Every clip under `root`, in name order.

    Name order rather than directory order so that two shards of the same run,
    or a re-run after a crash, walk the corpus the same way.
    """
    root = Path(root).expanduser().resolve()
    if is_clip(root):
        return [Clip(root)]
    found = sorted(p for p in root.glob(CLIP_GLOB) if is_clip(p))
    if limit is not None:
        found = found[:limit]
    return [Clip(p) for p in found]
