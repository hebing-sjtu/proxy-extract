"""Figure: does the low-poly render support the same extraction as the high one?

Pixel accuracy between the two label maps is a trap here — one or two classes
cover most of the frame, so a run that loses every small class still scores in
the nineties. This measures per-class IoU instead, and for depth measures the
relative difference per pixel, both aggregated over frames and clips.

The high render's own prediction is used as the reference. It is not ground
truth, but it is the answer the pipeline would get if the proxy were allowed to
see the good pixels, which is exactly the comparison the feasibility call needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import taxonomy, video
from proxy_extract.depth.depth_anything import DepthAnythingBackend
from proxy_extract.semantic.panoptic import PanopticBackend

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"
CACHE = ROOT / "experiments" / "feasibility.json"

SEG_SIZE = (640, 360)
DEPTH_SIZE = (644, 364)
FRAMES = [8, 32, 56, 80, 104]

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
    "17_michael_seg_0042",
    "48_franklin_seg_0100",
]


def match(source: np.ndarray, shape: tuple[int, int], nearest: bool) -> np.ndarray:
    if source.shape[:2] == shape:
        return source
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(source, (shape[1], shape[0]), interpolation=flags)


def collect() -> dict:
    semantic = PanopticBackend(batch_size=1)
    depth = DepthAnythingBackend(batch_size=1)

    intersection = np.zeros(taxonomy.NUM_CLASSES)
    union = np.zeros(taxonomy.NUM_CLASSES)
    reference_pixels = np.zeros(taxonomy.NUM_CLASSES)
    per_clip = []

    for stem in CLIPS:
        clip_inter = np.zeros(taxonomy.NUM_CLASSES)
        clip_union = np.zeros(taxonomy.NUM_CLASSES)
        depth_errors = []

        for frame_index in FRAMES:
            maps = {}
            for kind in ("high", "low"):
                path = DATA / kind / f"{stem}.mp4"
                colour = video.read_frames(path, size=SEG_SIZE, limit=frame_index + 1)[frame_index]
                for_depth = video.read_frames(path, size=DEPTH_SIZE, limit=frame_index + 1)[frame_index]
                maps[kind] = (
                    semantic.segment([colour]).labels[0],
                    depth.estimate([for_depth]).depth[0],
                )

            high_labels, high_depth = maps["high"]
            low_labels = match(maps["low"][0].astype(np.uint8), high_labels.shape, nearest=True)
            low_depth = match(maps["low"][1], high_depth.shape, nearest=False)

            for cls in range(taxonomy.NUM_CLASSES):
                a, b = high_labels == cls, low_labels == cls
                clip_inter[cls] += np.sum(a & b)
                clip_union[cls] += np.sum(a | b)
                reference_pixels[cls] += np.sum(a)

            # Relative rather than absolute difference: a 2 m disagreement at
            # 50 m is irrelevant to the log encoding, at 3 m it is not.
            valid = (high_depth > 1) & (high_depth < 79) & (low_depth > 1) & (low_depth < 79)
            depth_errors.append(
                float(np.median(np.abs(low_depth[valid] - high_depth[valid]) / high_depth[valid]))
            )

        intersection += clip_inter
        union += clip_union
        with np.errstate(divide="ignore", invalid="ignore"):
            clip_iou = np.where(clip_union > 0, clip_inter / clip_union, np.nan)
        per_clip.append(
            {
                "clip": stem,
                "miou": float(np.nanmean(clip_iou)),
                "depth_rel": float(np.median(depth_errors)),
            }
        )
        print(f"{stem:32} mIoU {per_clip[-1]['miou'] * 100:5.1f}%  depth {per_clip[-1]['depth_rel'] * 100:5.1f}%")

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, np.nan)
    return {
        "per_class_iou": iou.tolist(),
        "reference_pixels": reference_pixels.tolist(),
        "per_clip": per_clip,
    }


def plot(data: dict, path: Path) -> None:
    iou = np.array(data["per_class_iou"])
    share = np.array(data["reference_pixels"])
    share = share / share.sum()
    present = np.where(share > 1e-4)[0]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.4), width_ratios=[1.25, 1])

    order = present[np.argsort(-share[present])]
    x = np.arange(len(order))
    colours = ["#1f8a4c" if iou[c] > 0.6 else "#c8952b" if iou[c] > 0.35 else "#b3382c" for c in order]
    left.bar(x, iou[order] * 100, color=colours)
    for slot, cls in enumerate(order):
        left.text(slot, iou[cls] * 100 + 1.5, f"{share[cls] * 100:.0f}%", ha="center", fontsize=7, color="#555")
    left.set_xticks(x, [taxonomy.CLASS_NAMES[c] for c in order], rotation=35, ha="right", fontsize=8)
    left.set_ylabel("IoU between low-render and high-render labels (%)")
    left.set_ylim(0, 100)
    left.set_title(
        "Per-class agreement.\nGrey number = share of pixels the class covers in the high render.",
        fontsize=10,
    )
    left.grid(axis="y", alpha=0.25)

    records = sorted(data["per_clip"], key=lambda r: r["miou"])
    y = np.arange(len(records))
    right.barh(y - 0.2, [r["miou"] * 100 for r in records], height=0.38, color="#3b6fb6", label="semantic mIoU")
    right.barh(
        y + 0.2,
        [r["depth_rel"] * 100 for r in records],
        height=0.38,
        color="#d1791f",
        label="median relative depth difference",
    )
    right.set_yticks(y, [r["clip"].split("_seg_")[0] for r in records], fontsize=8)
    right.set_xlabel("percent")
    right.set_title("Per clip: semantics vary a lot, depth barely moves", fontsize=10)
    right.legend(fontsize=8, frameon=False, loc="lower right")
    right.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Low-poly render vs high render, same models, same frames: "
        "depth transfers, semantics only partly do",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(CACHE.read_text()) if CACHE.exists() else collect()
    if not CACHE.exists():
        CACHE.write_text(json.dumps(data))
    plot(data, OUT / "feasibility.png")
    print(f"wrote {OUT / 'feasibility.png'}")


if __name__ == "__main__":
    main()
