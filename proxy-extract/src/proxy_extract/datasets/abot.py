"""Reader for ABot-World-Explorer episodes.

Layout, per the dataset card (30,969 episodes, Apache-2.0):

    data/<prefix>/<sample_id>/video.mp4
    data/<prefix>/<sample_id>/annotations.tar

`annotations.tar` is an uncompressed POSIX USTAR holding `action.json`,
`caption.json` and a complete `sparse/0/{cameras,images,points3D}.txt` COLMAP
model. The card lists `Semantic splits: None` — there is no depth and no
semantic anywhere in this corpus, which is the whole reason the prediction
pipeline exists.

Everything is read straight out of the tar rather than unpacked. At this
episode count the extracted COLMAP text would cost far more inodes than it is
worth, and the members are small.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cameras import CameraTrack, from_colmap_text

ANNOTATION_MEMBERS = ("action.json", "caption.json")
SPARSE_MEMBERS = ("cameras.txt", "images.txt", "points3D.txt")


@dataclass(frozen=True)
class Episode:
    sample_id: str
    video: Path
    annotations: Path

    def exists(self) -> bool:
        return self.video.is_file() and self.annotations.is_file()


def discover(root: Path) -> list[Episode]:
    """Find every episode under a snapshot download of the dataset."""
    root = Path(root)
    data = root / "data" if (root / "data").is_dir() else root
    episodes = []
    for video in sorted(data.glob("*/*/video.mp4")):
        episodes.append(
            Episode(
                sample_id=video.parent.name,
                video=video,
                annotations=video.parent / "annotations.tar",
            )
        )
    return episodes


def read_members(annotations: Path) -> dict[str, bytes]:
    """Pull the members this package cares about out of `annotations.tar`.

    Members are matched by basename because the card does not pin the leading
    directory, and a path-prefix assumption would break silently on a repack.
    """
    wanted = set(ANNOTATION_MEMBERS) | set(SPARSE_MEMBERS)
    found: dict[str, bytes] = {}
    with tarfile.open(annotations, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name in wanted and name not in found:
                handle = archive.extractfile(member)
                if handle is not None:
                    found[name] = handle.read()
    return found


def load_actions(annotations: Path) -> list[dict]:
    """Per-frame keyboard actions.

    Worth more here than the label they look like: whether the player pressed
    forward is a fact about the protagonist that no bystander shares, so this
    is an independent signal for the player/ped split. Nothing scores it yet —
    the corpus ships no player/ped ground truth — so anything built on it has
    to be validated separately before it can be trusted.
    """
    payload = json.loads(read_members(annotations)["action.json"])
    if isinstance(payload, dict):
        for key in ("actions", "frames", "data"):
            if key in payload:
                return list(payload[key])
        return [payload]
    return list(payload)


def load_caption(annotations: Path) -> dict:
    return json.loads(read_members(annotations)["caption.json"])


def load_cameras(annotations: Path, *, scratch: Path | None = None) -> CameraTrack:
    """The episode's COLMAP model as a CameraTrack, `metric=False`.

    COLMAP's own reader wants files on disk, so the three text members are
    written to a scratch directory first. Non-metric always: a sparse
    reconstruction pins the scene only up to a similarity.
    """
    import tempfile

    members = read_members(annotations)
    missing = [name for name in ("cameras.txt", "images.txt") if name not in members]
    if missing:
        raise KeyError(f"{annotations} has no {missing}; not a complete COLMAP model")

    context = tempfile.TemporaryDirectory() if scratch is None else None
    target = Path(context.name) if context is not None else Path(scratch)
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in SPARSE_MEMBERS:
            if name in members:
                (target / name).write_bytes(members[name])
        return from_colmap_text(target)
    finally:
        if context is not None:
            context.cleanup()


def load_points(annotations: Path) -> np.ndarray:
    """The (M, 3) sparse cloud, or an empty array if the episode has none."""
    from ..cameras import read_colmap_points

    import tempfile

    members = read_members(annotations)
    if "points3D.txt" not in members:
        return np.zeros((0, 3), dtype=np.float64)
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "points3D.txt"
        path.write_bytes(members["points3D.txt"])
        return read_colmap_points(path)
