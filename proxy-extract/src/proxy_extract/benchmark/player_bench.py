"""Score the player/ped split against the engine's own labels.

The trick that makes this cheap: gta-web labels `player` and `ped` separately,
so merging the two classes reconstructs exactly what a segmenter would hand the
splitter, and the labels that were merged away are the answer key. No model, no
GPU, no annotation — one pass over the GT semantic videos.

Two things come out of it.

`measure_anchor` reports where the protagonist actually sits on screen, across
however many clips you point it at. `PlayerPrior.anchor` is currently a guess
that third-person rigs frame the character low-centre; this replaces the guess
with the distribution, and `max_anchor_distance` should be read off its tail
rather than picked.

`score_clip` then runs the real splitter on the merged input and grades it. The
breakdown by tag is the point rather than the headline number: `drive` clips
put the protagonist inside a car where the standard gives ego and traffic the
same id, and `look` clips orbit a stationary character. Those are different
problems, and an average over all four hides both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..semantic.player import PlayerPrior, split, with_anchor
from ..taxonomy import S11_PED, S11_PLAYER

# Above this the predicted player mask is the right person. Set well below 1.0
# because the splitter promotes whole connected components while the GT traces
# the character exactly, so even a perfect decision loses the boundary pixels.
CORRECT_IOU = 0.5


@dataclass(frozen=True)
class ClipScore:
    tag: str
    frames: int
    player_present: bool
    decided: bool
    correct: bool
    iou: float
    note: str

    @property
    def declined(self) -> bool:
        return not self.decided


def merge_people(truth: np.ndarray) -> np.ndarray:
    """Collapse `player` into `ped`, reproducing a segmenter's undivided output."""
    merged = np.asarray(truth, dtype=np.uint8).copy()
    merged[merged == S11_PLAYER] = S11_PED
    return merged


def player_centroids(truth: np.ndarray) -> np.ndarray:
    """(frames, 2) normalised (x, y) of the GT player, NaN where absent."""
    truth = np.asarray(truth)
    height, width = truth.shape[1:]
    out = np.full((len(truth), 2), np.nan)
    for index, frame in enumerate(truth):
        ys, xs = np.nonzero(frame == S11_PLAYER)
        if xs.size:
            out[index] = (xs.mean() / width, ys.mean() / height)
    return out


def measure_anchor(clips: list[np.ndarray]) -> dict:
    """Where the protagonist actually is on screen, over a set of GT clips.

    The output feeds `PlayerPrior` directly: the median is the anchor, and the
    95th percentile of the radius is the widest `max_anchor_distance` that
    still admits real players.
    """
    centroids = np.concatenate([player_centroids(clip) for clip in clips]) if clips else np.empty((0, 2))
    visible = centroids[~np.isnan(centroids).any(axis=1)]
    if not visible.size:
        return {"frames_with_player": 0}

    anchor = (float(np.median(visible[:, 0])), float(np.median(visible[:, 1])))
    radius = np.linalg.norm(visible - np.array(anchor), axis=1)
    return {
        "frames_with_player": int(len(visible)),
        "frames_total": int(len(centroids)),
        "visible_fraction": round(float(len(visible) / max(len(centroids), 1)), 4),
        "anchor": (round(anchor[0], 4), round(anchor[1], 4)),
        "radius_p50": round(float(np.percentile(radius, 50)), 4),
        "radius_p90": round(float(np.percentile(radius, 90)), 4),
        "radius_p95": round(float(np.percentile(radius, 95)), 4),
        "x_iqr": [round(float(np.percentile(visible[:, 0], 25)), 4), round(float(np.percentile(visible[:, 0], 75)), 4)],
        "y_iqr": [round(float(np.percentile(visible[:, 1], 25)), 4), round(float(np.percentile(visible[:, 1], 75)), 4)],
    }


def score_clip(truth: np.ndarray, *, tag: str = "", prior: PlayerPrior | None = None) -> ClipScore:
    """Run the splitter on the merged GT and grade it against the real labels."""
    truth = np.asarray(truth, dtype=np.uint8)
    gt_player = truth == S11_PLAYER
    present = bool(gt_player.any())

    result = split(merge_people(truth), prior)
    predicted = result.labels == S11_PLAYER

    intersection = int(np.logical_and(predicted, gt_player).sum())
    union = int(np.logical_or(predicted, gt_player).sum())
    iou = float(intersection / union) if union else float("nan")

    # Declining on a clip with no player is the right answer, not a miss.
    if not present:
        correct = not result.resolved
    else:
        correct = result.resolved and iou >= CORRECT_IOU

    return ClipScore(
        tag=tag,
        frames=len(truth),
        player_present=present,
        decided=result.resolved,
        correct=correct,
        iou=iou,
        note=result.note,
    )


def aggregate(scores: list[ClipScore]) -> dict:
    """Roll clip scores up overall and per tag.

    Decline rate is reported next to accuracy rather than folded into it. A
    splitter that answers half as often and is right when it does is a
    different proposition from one that always answers and is sometimes wrong,
    and only one of those is safe to put in a dataset.
    """
    if not scores:
        return {"clips": 0}

    def summarise(group: list[ClipScore]) -> dict:
        with_player = [s for s in group if s.player_present]
        decided = [s for s in group if s.decided]
        ious = [s.iou for s in decided if s.player_present and np.isfinite(s.iou)]
        return {
            "clips": len(group),
            "clips_with_player": len(with_player),
            "decided_fraction": round(len(decided) / len(group), 4),
            "accuracy": round(sum(s.correct for s in group) / len(group), 4),
            "accuracy_when_decided": (
                round(sum(s.correct for s in decided) / len(decided), 4) if decided else None
            ),
            "mean_iou_when_decided": round(float(np.mean(ious)), 4) if ious else None,
        }

    tags = sorted({s.tag for s in scores if s.tag})
    return {
        "overall": summarise(scores),
        "by_tag": {tag: summarise([s for s in scores if s.tag == tag]) for tag in tags},
    }


def sweep_anchor(
    clips: list[tuple[np.ndarray, str]],
    anchors: list[tuple[float, float]],
    *,
    prior: PlayerPrior | None = None,
) -> list[dict]:
    """Accuracy as a function of the assumed screen anchor.

    Run this before trusting the default. If the curve is flat the prior is
    doing no work and something simpler would do; if it peaks sharply, the peak
    is the number to ship.
    """
    base = prior or PlayerPrior()
    results = []
    for anchor in anchors:
        candidate = with_anchor(base, anchor)
        scores = [score_clip(truth, tag=tag, prior=candidate) for truth, tag in clips]
        summary = aggregate(scores)
        results.append({"anchor": anchor, **summary["overall"]})
    return results
