"""Figure: how many metres is one unit of the supplied camera track?

COLMAP recovers geometry only up to a similarity, so the translations in
`camera/*.json` carry no physical unit. DUV, however, demands metres. This
measures the missing factor per clip by triangulating sparse correspondences
with the GT poses (giving depth in world units) and reading a metric monocular
model at the same pixels (giving metres), then taking the robust ratio.

If that factor is close to 1 and stable across clips, the tracks can be treated
as metric and the whole calibration stage collapses to a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import camera_qc, cameras, video
from proxy_extract.depth.depth_anything import DepthAnythingBackend

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"
CACHE = ROOT / "experiments" / "scale.json"

# Depth Anything ingests 14-pixel patches, and these clips are 16:9.
INFER_SIZE = (644, 364)
GAP = 8
STRIDE = 16


def clip_scale(stem: str, backend: DepthAnythingBackend) -> dict:
    track = cameras.load(DATA / "camera" / f"{stem}.json")
    colour = video.read_frames(DATA / "high" / f"{stem}.mp4", size=INFER_SIZE)
    gray = video.read_frames(DATA / "high" / f"{stem}.mp4", grayscale=True)

    scene_scale = float(np.linalg.norm(track.positions.max(axis=0) - track.positions.min(axis=0)))
    indices = list(range(0, len(gray) - GAP, STRIDE))
    depth = backend.estimate([colour[i] for i in indices]).depth

    ratios, world_depths, metric_depths = [], [], []
    for slot, index in enumerate(indices):
        evidence = camera_qc.evaluate_pair(
            gray[index], gray[index + GAP], track, index, index + GAP, scene_scale
        )
        if evidence.degenerate or len(evidence.depths) < camera_qc.MIN_TRACKED_POINTS:
            continue

        # Correspondences live in 1280x720; the depth map is at INFER_SIZE.
        columns = np.clip(
            (evidence.pixels[:, 0] * INFER_SIZE[0] / 1280).astype(int), 0, INFER_SIZE[0] - 1
        )
        rows = np.clip((evidence.pixels[:, 1] * INFER_SIZE[1] / 720).astype(int), 0, INFER_SIZE[1] - 1)
        sampled = depth[slot][rows, columns]

        # Depth Anything's outdoor head saturates near its 80 m training
        # ceiling, and the flat top of that curve would drag the ratio down, so
        # only the range where it is actually resolving depth is used.
        usable = (sampled > 2.0) & (sampled < 40.0) & (evidence.depths > 0)
        if usable.sum() < camera_qc.MIN_TRACKED_POINTS:
            continue
        world_depths.append(evidence.depths[usable])
        metric_depths.append(sampled[usable])
        ratios.append(np.median(sampled[usable] / evidence.depths[usable]))

    if not ratios:
        return {"clip": stem, "metres_per_unit": float("nan"), "spread": float("nan"), "pairs": 0}

    ratios = np.array(ratios)
    return {
        "clip": stem,
        "metres_per_unit": float(np.median(ratios)),
        # Interquartile spread relative to the median: how much the answer
        # wanders between frame pairs within one clip.
        "spread": float((np.percentile(ratios, 75) - np.percentile(ratios, 25)) / np.median(ratios)),
        "pairs": len(ratios),
        "world_depths": np.concatenate(world_depths).tolist(),
        "metric_depths": np.concatenate(metric_depths).tolist(),
    }


CLIPS = [
    "02_john_marston_seg_0017",
    "11_john_marston_seg_0313",
    "12_john_marston_seg_0341",
    "26_trevor_seg_0004",
    "29_trevor_seg_0063",
    "32_trevor_seg_0085",
    "33_trevor_seg_0096",
    "39_franklin_seg_0041",
    "42_franklin_seg_0067",
    "46_franklin_seg_0093",
]


def plot(records: list[dict], path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), width_ratios=[1.15, 1])

    for record in records:
        if not record.get("world_depths"):
            continue
        left.scatter(
            record["world_depths"],
            record["metric_depths"],
            s=3,
            alpha=0.25,
            label=record["clip"].split("_seg_")[0],
        )

    limits = np.array([0.5, 120])
    left.plot(limits, limits, color="black", lw=1.2, ls="--", label="1 unit = 1 m")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(*limits)
    left.set_ylim(*limits)
    left.set_xlabel("triangulated depth using GT poses (world units)")
    left.set_ylabel("Depth Anything V2 metric depth (m)")
    left.set_title("Every tracked point, both ways of measuring its depth")
    left.grid(alpha=0.25, which="both")
    left.legend(fontsize=7, markerscale=3, ncol=2, loc="upper left", frameon=False)

    scored = [r for r in records if np.isfinite(r["metres_per_unit"])]
    order = sorted(scored, key=lambda r: r["metres_per_unit"])
    y = np.arange(len(order))
    values = np.array([r["metres_per_unit"] for r in order])
    spreads = np.array([r["spread"] for r in order])

    right.barh(y, values, xerr=values * spreads / 2, color="#3b6fb6", height=0.6, capsize=3)
    right.axvline(1.0, color="black", ls="--", lw=1.2)
    right.set_yticks(y, [r["clip"].split("_seg_")[0] for r in order], fontsize=8)
    right.set_xlabel("metres per world unit")
    right.set_title(
        f"Per-clip scale factor (median {np.median(values):.2f},"
        f" spread {values.min():.2f}-{values.max():.2f})"
    )
    right.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "The GT camera tracks are already close to metric, but not exactly, and not consistently",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        records = json.loads(CACHE.read_text())
    else:
        backend = DepthAnythingBackend(batch_size=4)
        records = []
        for stem in CLIPS:
            record = clip_scale(stem, backend)
            records.append(record)
            print(
                f"{stem:32} {record['metres_per_unit']:6.3f} m/unit "
                f"spread {record['spread']:5.2f} over {record['pairs']} pairs"
            )
        CACHE.write_text(json.dumps(records))

    plot(records, OUT / "scale_calibration.png")
    print(f"wrote {OUT / 'scale_calibration.png'}")


if __name__ == "__main__":
    main()
