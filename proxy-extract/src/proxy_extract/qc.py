"""Screen high/low clip pairs for geometric drift before they enter the dataset.

The low-poly track is an AI restyle of the source, not a re-render of the same
3D scene, so it is free to reinvent geometry. On the 29-clip handpick sample
roughly a quarter of pairs drift far enough that the proxy no longer describes
the target. Those pairs teach the v2v model to hallucinate offsets, so they have
to be caught here rather than discovered as training noise.

Appearance cannot be compared directly across such different art styles. What
can be compared is each clip's *own* optical flow: flow is induced by camera
motion and scene depth, so a restyle that preserved geometry and camera must
reproduce it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contract import CONDITION_HEIGHT, CONDITION_WIDTH
from .video import read_frames

# Thresholds on EPE/motion, calibrated against visual inspection of the 29-clip
# handpick sample. See docs in README for the montage those were read off.
KEEP_MAX_EPE_REL = 0.22
REVIEW_MAX_EPE_REL = 0.45

FLOW_GAP_FRAMES = 4
FLOW_SAMPLES = 10


@dataclass(frozen=True)
class AlignmentReport:
    clip: str
    tier: str
    flow_cos: float
    epe_rel: float
    epe_px: float
    motion_px: float
    mag_ratio: float
    samples: int

    @property
    def usable(self) -> bool:
        return self.tier != "drop"


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 5, 21, 3, 7, 1.5, 0)


def tier_for(epe_rel: float) -> str:
    """Grade a pair from its normalised flow disagreement.

    EPE/motion leads and cosine only annotates: flat, low-texture scenes (open
    pavement, sky) produce near-random flow directions, which tanks cosine even
    when the two clips are in excellent agreement. `26_trevor_seg_0004` is the
    canonical example - cos 0.803 but EPE/motion 0.209, and visibly well aligned.
    """
    if epe_rel <= KEEP_MAX_EPE_REL:
        return "keep"
    if epe_rel <= REVIEW_MAX_EPE_REL:
        return "review"
    return "drop"


def score_pair(
    high_path: Path,
    low_path: Path,
    *,
    clip_id: str | None = None,
    gap: int = FLOW_GAP_FRAMES,
    samples: int = FLOW_SAMPLES,
) -> AlignmentReport:
    """Compare the two tracks' internal flow fields at condition resolution."""
    size = (CONDITION_WIDTH, CONDITION_HEIGHT)
    high = read_frames(high_path, size=size, grayscale=True)
    low = read_frames(low_path, size=size, grayscale=True)

    n = min(len(high), len(low))
    if n <= gap:
        raise ValueError(f"need more than {gap} frames to measure flow, got {n}")
    starts = np.unique(np.linspace(0, n - 1 - gap, samples).astype(int))

    cos_all, ratio_all, epe_all, mag_all = [], [], [], []
    for t in starts:
        vh, vl = _flow(high[t], high[t + gap]), _flow(low[t], low[t + gap])
        mh, ml = np.linalg.norm(vh, axis=2), np.linalg.norm(vl, axis=2)
        # Score only where the source actually moves; static pixels carry no
        # geometric information and their flow is pure noise.
        moving = mh > max(0.5, float(np.quantile(mh, 0.5)))
        if int(moving.sum()) < 200:
            continue
        cos_all.append(float(np.median(((vh * vl).sum(axis=2) / (mh * ml + 1e-6))[moving])))
        ratio_all.append(float(np.median(ml[moving] / (mh[moving] + 1e-6))))
        epe_all.append(float(np.median(np.linalg.norm(vh - vl, axis=2)[moving])))
        mag_all.append(float(np.median(mh[moving])))

    if not epe_all:
        raise ValueError(f"no frame pair in {high_path.name} had enough motion to score")

    epe_px = float(np.mean(epe_all))
    motion_px = float(np.mean(mag_all))
    epe_rel = epe_px / max(motion_px, 1e-6)
    return AlignmentReport(
        clip=clip_id or high_path.stem,
        tier=tier_for(epe_rel),
        flow_cos=round(float(np.mean(cos_all)), 4),
        epe_rel=round(epe_rel, 4),
        epe_px=round(epe_px, 3),
        motion_px=round(motion_px, 3),
        mag_ratio=round(float(np.mean(ratio_all)), 4),
        samples=len(epe_all),
    )


def score_dataset(root: Path, *, high_dir: str = "high", low_dir: str = "low") -> list[AlignmentReport]:
    """Score every clip that exists in both the high and low directories."""
    root = Path(root)
    highs = {p.name: p for p in sorted((root / high_dir).glob("*.mp4"))}
    lows = {p.name: p for p in sorted((root / low_dir).glob("*.mp4"))}
    shared = sorted(set(highs) & set(lows))
    if not shared:
        raise ValueError(f"no clip name appears in both {root / high_dir} and {root / low_dir}")
    return [score_pair(highs[name], lows[name], clip_id=Path(name).stem) for name in shared]


def write_report(reports: list[AlignmentReport], path: Path) -> dict:
    """Persist per-clip scores plus a tier summary."""
    ordered = sorted(reports, key=lambda r: r.epe_rel)
    summary = {tier: sum(1 for r in ordered if r.tier == tier) for tier in ("keep", "review", "drop")}
    payload = {
        "thresholds": {"keep_max_epe_rel": KEEP_MAX_EPE_REL, "review_max_epe_rel": REVIEW_MAX_EPE_REL},
        "summary": summary,
        "clips": [asdict(r) for r in ordered],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload
