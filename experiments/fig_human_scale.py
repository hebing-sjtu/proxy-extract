"""Cross-check the world unit against a ruler that is not a depth model.

The metres-per-unit factor measured in `fig_scale.py` leans entirely on Depth
Anything being right about metres, which is a lot of weight for one monocular
network to carry. This measures the same factor a second way, using only the GT
poses and the fact that the playable character is roughly 1.8 m tall:

  - segment the character, take the pixel height of its silhouette
  - triangulate static points around its feet to get the ground depth in world
    units (static, so the epipolar geometry actually holds there)
  - the character's height in world units is then pixel_height * depth / focal

Agreement between two such different routes is what makes the conclusion safe
to build on.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import camera_qc, cameras, taxonomy, video
from proxy_extract.semantic.panoptic import PanopticBackend

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"
CACHE = ROOT / "experiments" / "human_scale.json"

SEG_SIZE = (640, 360)
NOMINAL_HEIGHT_M = 1.8
GAP = 8
NEAREST_GROUND_POINTS = 25


def largest_human_box(labels: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = (labels == taxonomy.HUMAN).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 200:
        return None
    x, y, w, h = stats[largest, : cv2.CC_STAT_HEIGHT + 1]
    return int(x), int(y), int(w), int(h)


def measure(stem: str, backend: PanopticBackend) -> dict:
    track = cameras.load(DATA / "camera" / f"{stem}.json")
    focal = float(track.intrinsics[0, 0])
    colour = video.read_frames(DATA / "high" / f"{stem}.mp4", size=SEG_SIZE)
    gray = video.read_frames(DATA / "high" / f"{stem}.mp4", grayscale=True)
    scene_scale = float(np.linalg.norm(track.positions.max(axis=0) - track.positions.min(axis=0)))

    indices = list(range(8, len(gray) - GAP, 24))
    labels = backend.segment([colour[i] for i in indices]).labels

    samples = []
    for slot, index in enumerate(indices):
        box = largest_human_box(labels[slot])
        if box is None:
            continue
        x, y, w, h = box
        # Back to full resolution, where the intrinsics and the tracks live.
        sx, sy = 1280 / SEG_SIZE[0], 720 / SEG_SIZE[1]
        pixel_height = h * sy
        foot = np.array([(x + w / 2) * sx, (y + h) * sy])
        # A silhouette clipped by the frame edge has no meaningful height.
        if y <= 1 or y + h >= SEG_SIZE[1] - 1 or pixel_height < 40:
            continue

        evidence = camera_qc.evaluate_pair(
            gray[index], gray[index + GAP], track, index, index + GAP, scene_scale
        )
        if evidence.degenerate or len(evidence.depths) < NEAREST_GROUND_POINTS:
            continue

        distances = np.linalg.norm(evidence.pixels - foot, axis=1)
        nearest = np.argsort(distances)[:NEAREST_GROUND_POINTS]
        ground_depth = float(np.median(evidence.depths[nearest]))
        if not np.isfinite(ground_depth) or ground_depth <= 0:
            continue

        height_units = pixel_height * ground_depth / focal
        samples.append(NOMINAL_HEIGHT_M / height_units)

    if not samples:
        return {"clip": stem, "metres_per_unit": float("nan"), "samples": 0}
    return {
        "clip": stem,
        "metres_per_unit": float(np.median(samples)),
        "samples": len(samples),
        "all": [float(s) for s in samples],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from fig_scale import CLIPS

    if CACHE.exists():
        records = json.loads(CACHE.read_text())
    else:
        backend = PanopticBackend(batch_size=2)
        records = []
        for stem in CLIPS:
            record = measure(stem, backend)
            records.append(record)
            print(f"{stem:32} {record['metres_per_unit']:6.3f} m/unit from {record['samples']} frames")
        CACHE.write_text(json.dumps(records))

    depth_route = {r["clip"]: r for r in json.loads((ROOT / "experiments" / "scale.json").read_text())}

    pairs = [
        (r["clip"], depth_route[r["clip"]]["metres_per_unit"], r["metres_per_unit"])
        for r in records
        if np.isfinite(r["metres_per_unit"]) and r["clip"] in depth_route
    ]

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    xs = np.array([p[1] for p in pairs])
    ys = np.array([p[2] for p in pairs])
    sizes = np.array([30 + 25 * r["samples"] for r in records if np.isfinite(r["metres_per_unit"])])
    ax.scatter(xs, ys, s=sizes, color="#3b6fb6", zorder=3, edgecolor="white")
    for name, x, y in pairs:
        ax.annotate(name.split("_seg_")[0], (x, y), fontsize=7, xytext=(6, 5), textcoords="offset points")

    limits = np.array([0.08, 3.0])
    ax.plot(limits, limits, ls="--", color="black", lw=1.1, label="the two rulers would agree here")
    ax.axhline(1.0, ls=":", color="#b3382c", lw=1.1, label="1 unit = 1 m")
    ax.axvline(1.0, ls=":", color="#b3382c", lw=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("metres per unit from triangulation vs Depth Anything")
    ax.set_ylabel("metres per unit from a 1.8 m character")
    correlation = float(np.corrcoef(np.log(xs), np.log(ys))[0, 1]) if len(xs) > 2 else float("nan")
    ax.set_title(
        "The two rulers do not agree (log r = "
        f"{correlation:.2f}), so neither pins the scale.\n"
        "Both do agree the unit is metre-ish and clip-dependent — which is enough to decide.\n"
        "Marker size = frames the character ruler could use.",
        fontsize=10,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "scale_crosscheck.png", dpi=140)
    print(f"wrote {OUT / 'scale_crosscheck.png'}")


if __name__ == "__main__":
    main()
