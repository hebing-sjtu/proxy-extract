"""Promote the protagonist out of the generic person class.

Both person-splitting taxonomies need the same orchestration around two
different splitters - coarse6 calls the pair hero/npc, standard11 calls it
player/ped - and both the condition_root pipeline and the 720p delivery
pipeline need that orchestration. It lives here so the two cannot drift into
reporting the split differently.
"""

from __future__ import annotations

import numpy as np


def split_people(labels: np.ndarray, *, taxonomy: str, enabled: bool = True) -> tuple[np.ndarray, dict]:
    """Relabel the protagonist's pixels, returning the labels and a diagnosis.

    Call this *after* temporal stabilisation, so the tracker sees settled masks
    rather than chasing per-frame flicker into spurious tracks.

    The diagnosis is not optional colour. Both splitters decline when the
    evidence is ambiguous, and a caller that cannot tell "no protagonist here"
    from "we gave up" would silently ship clips where every person is a
    bystander.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    if not enabled:
        return labels, {"attempted": False}

    if taxonomy == "coarse6":
        from .hero import split as split_hero

        result = split_hero(labels)
        return result.labels, {
            "attempted": True,
            "resolved": result.hero_track is not None,
            "note": result.note,
            "person_tracks": len(result.tracks),
            "multi_person_frames": round(result.multi_person_frames, 4),
            "merged_frames": round(result.merged_frames, 4),
        }

    if taxonomy == "standard11":
        from .player import split as split_player

        result = split_player(labels)
        return result.labels, {
            "attempted": True,
            "resolved": result.resolved,
            "note": result.note,
            "person_tracks": len(result.tracks),
            "multi_person_frames": round(result.multi_person_frames, 4),
            "merged_frames": round(result.merged_frames, 4),
            "driving": result.driving,
        }

    # cwm12 has no protagonist class to promote into.
    return labels, {"attempted": False}
