"""Figure: what the semantic and depth stages actually produce, high vs low.

This is the panel the feasibility call rests on. For each clip it shows the two
renders and, underneath each, the CWM-taxonomy labels and the metric depth a
real model returns for it. The question it answers is not "is the model good"
but "does the low-poly render carry enough signal to get the same answer as the
high render", because at inference time only the low side exists.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import taxonomy, video
from proxy_extract.depth.depth_anything import DepthAnythingBackend
from proxy_extract.preview import CLASS_COLORS, colorize_semantic
from proxy_extract.semantic.panoptic import PanopticBackend

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"

SEG_SIZE = (640, 360)
DEPTH_SIZE = (644, 364)
FRAME = 40

CLIPS = [
    ("11_john_marston_seg_0313", "epipolar 0.29 px"),
    ("26_trevor_seg_0004", "epipolar 0.65 px"),
    ("46_franklin_seg_0093", "epipolar 1.73 px"),
    ("17_michael_seg_0042", "epipolar 4.52 px"),
]


def mean_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Mean IoU over the classes either render used, after size matching.

    Not pixel accuracy: terrain and road alone cover most of these frames, so
    accuracy stays in the nineties even when every small class is lost.
    """
    import cv2

    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    scores = []
    for cls in np.union1d(np.unique(a), np.unique(b)):
        hit, miss = (a == cls), (b == cls)
        scores.append(np.sum(hit & miss) / np.sum(hit | miss))
    return float(np.mean(scores))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    semantic = PanopticBackend(batch_size=1)
    depth = DepthAnythingBackend(batch_size=1)

    rows = 3
    fig, axes = plt.subplots(
        rows * 2, len(CLIPS), figsize=(3.6 * len(CLIPS), 2.15 * rows * 2), constrained_layout=True
    )
    used_classes: set[int] = set()
    scores = []

    for column, (stem, note) in enumerate(CLIPS):
        labels_by_kind = {}
        for kind_index, kind in enumerate(("high", "low")):
            path = DATA / kind / f"{stem}.mp4"
            colour = video.read_frames(path, size=SEG_SIZE, limit=FRAME + 1)[FRAME]
            for_depth = video.read_frames(path, size=DEPTH_SIZE, limit=FRAME + 1)[FRAME]

            labels = semantic.segment([colour]).labels[0]
            metres = depth.estimate([for_depth]).depth[0]
            labels_by_kind[kind] = labels
            used_classes.update(np.unique(labels).tolist())

            base = kind_index * rows
            axes[base + 0, column].imshow(colour)
            axes[base + 0, column].set_title(f"{stem.split('_seg_')[0]} [{kind}]  {note}", fontsize=8)

            axes[base + 1, column].imshow(colorize_semantic(labels))
            axes[base + 1, column].set_title("semantic -> CWM 12 classes", fontsize=8)

            shown = axes[base + 2, column].imshow(
                np.clip(metres, 1, 60), cmap="turbo_r", norm=matplotlib.colors.LogNorm(1, 60)
            )
            axes[base + 2, column].set_title(
                f"metric depth  p5 {np.percentile(metres, 5):.1f}m"
                f"  p95 {np.percentile(metres, 95):.1f}m",
                fontsize=8,
            )
            if column == len(CLIPS) - 1 and kind == "low":
                fig.colorbar(shown, ax=axes[:, column], fraction=0.02, pad=0.01, label="metres")

        scores.append((stem, mean_iou(labels_by_kind["high"], labels_by_kind["low"])))

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=taxonomy.CLASS_NAMES[i])
        for i in sorted(used_classes)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)

    summary = "   ".join(f"{name.split('_seg_')[0]} {value * 100:.0f}%" for name, value in scores)
    fig.suptitle(
        "Real semantic and depth extraction, high render (rows 1-3) vs low-poly render (rows 4-6)\n"
        f"semantic mIoU between the two renders:   {summary}",
        fontsize=11,
    )
    fig.savefig(OUT / "extraction_high_vs_low.png", dpi=125, bbox_inches="tight")
    print(f"wrote {OUT / 'extraction_high_vs_low.png'}")
    for name, value in scores:
        print(f"{name:32} label agreement {value * 100:5.1f}%")


if __name__ == "__main__":
    main()
