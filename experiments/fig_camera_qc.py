"""Figures: are the supplied COLMAP camera tracks usable, and on which videos?

Renders two panels:
  camera_qc_overview.png  per-clip epipolar residual, high render vs low render
  camera_qc_detail.png    what a passing and a failing clip actually look like
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import camera_qc, cameras, video

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"
LOW_SIZE = (1344, 768)

MATCH_PX, LOOSE_PX = 1.0, 3.0


def low_intrinsics(track: cameras.CameraTrack) -> np.ndarray:
    """Retarget the 1280x720 intrinsics onto the 1344x768 low render.

    Which FOV convention the low render used is not recoverable from the data
    (all four candidates score within noise of each other), so we take the
    same-vertical-FOV reading and note that the choice does not matter at this
    residual level.
    """
    focal = track.intrinsics[0, 0] * LOW_SIZE[1] / 720.0
    return np.array([[focal, 0.0, LOW_SIZE[0] / 2], [0.0, focal, LOW_SIZE[1] / 2], [0.0, 0.0, 1.0]])


def measure_all() -> list[dict]:
    records = []
    for json_path in sorted((DATA / "camera").glob("*.json")):
        stem = json_path.stem
        track = cameras.load(json_path)
        low_track = cameras.CameraTrack(track.cam2world, low_intrinsics(track), metric=False)

        high = camera_qc.verify_track(
            video.read_frames(DATA / "high" / f"{stem}.mp4", grayscale=True), track, stem
        )
        low = camera_qc.verify_track(
            video.read_frames(DATA / "low" / f"{stem}.mp4", grayscale=True), low_track, stem
        )
        records.append(
            {
                "clip": stem,
                "high_sampson": high.sampson_median,
                "low_sampson": low.sampson_median,
                "high_inlier": high.inlier_fraction,
                "low_inlier": low.inlier_fraction,
                "high_tier": high.tier,
                "low_tier": low.tier,
            }
        )
        print(f"{stem:32} high={high.sampson_median:6.2f}px  low={low.sampson_median:6.2f}px")
    return records


def plot_overview(records: list[dict], path: Path) -> None:
    order = sorted(records, key=lambda r: r["low_sampson"])
    labels = [r["clip"].replace("_seg_", " ").replace("_", " ") for r in order]
    high = np.array([r["high_sampson"] for r in order])
    low = np.array([r["low_sampson"] for r in order])
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axvspan(0, MATCH_PX, color="#1f8a4c", alpha=0.10)
    ax.axvspan(MATCH_PX, LOOSE_PX, color="#c8952b", alpha=0.10)
    ax.axvspan(LOOSE_PX, 100, color="#b3382c", alpha=0.10)

    ax.barh(y + 0.20, high, height=0.38, color="#3b6fb6", label="high render")
    ax.barh(y - 0.20, low, height=0.38, color="#d1791f", label="low-poly render")

    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 20)
    ax.set_xlabel("median Sampson epipolar residual against GT cameras (pixels, log scale)")
    ax.set_title(
        "GT cameras describe the high render almost exactly;\n"
        "how well they describe the low-poly render is what varies",
        fontsize=12,
    )
    for x, name in ((MATCH_PX, "1 px"), (LOOSE_PX, "3 px")):
        ax.axvline(x, color="#444", lw=0.8, ls="--")
        ax.text(x, len(order) - 0.2, f" {name}", fontsize=8, color="#444", va="top")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_correspondences(ax, frame: np.ndarray, evidence: camera_qc.PairEvidence, fundamental, title: str):
    points_a, points_b = camera_qc.track_points(frame[0], frame[1])
    errors = camera_qc.sampson_distance(points_a, points_b, fundamental)
    ax.imshow(frame[0], cmap="gray", vmin=0, vmax=255)

    flow = points_b - points_a
    finite = np.isfinite(errors)
    scatter = ax.quiver(
        points_a[finite, 0],
        points_a[finite, 1],
        flow[finite, 0],
        flow[finite, 1],
        np.clip(errors[finite], 0, 8),
        cmap="turbo",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        clim=(0, 8),
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def plot_detail(records: list[dict], path: Path) -> None:
    ranked = sorted(records, key=lambda r: r["low_sampson"])
    chosen = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.4))
    mappable = None
    for column, record in enumerate(chosen):
        stem = record["clip"]
        track = cameras.load(DATA / "camera" / f"{stem}.json")
        index_a, index_b = 40, 48
        rotation, translation = cameras.relative_pose(track, index_a, index_b)

        for row, kind in enumerate(("high", "low")):
            frames = video.read_frames(DATA / kind / f"{stem}.mp4", grayscale=True)
            intrinsics = track.intrinsics if kind == "high" else low_intrinsics(track)
            fundamental = camera_qc.fundamental_from_pose(intrinsics, rotation, translation)
            mappable = draw_correspondences(
                axes[row, column],
                (frames[index_a], frames[index_b]),
                None,
                fundamental,
                f"{stem}  [{kind}]   {record[kind + '_sampson']:.2f} px",
            )

    fig.suptitle(
        "Sparse tracks over 8 frames, coloured by distance from the epipolar line the GT camera predicts\n"
        "(top: high render, bottom: low-poly render — same camera, same frames)",
        fontsize=11,
    )
    bar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.01)
    bar.set_label("Sampson residual (px)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT.parent / "camera_qc.json"
    if cache.exists():
        records = json.loads(cache.read_text())
    else:
        records = measure_all()
        cache.write_text(json.dumps(records, indent=1))

    plot_overview(records, OUT / "camera_qc_overview.png")
    plot_detail(records, OUT / "camera_qc_detail.png")
    print(f"wrote {OUT}/camera_qc_overview.png and camera_qc_detail.png")


if __name__ == "__main__":
    main()
