"""Figure: what the low-poly render actually looks like next to the high render.

This is the panel that decides which of semantics and depth is realistic to
extract from the low side, so it is worth looking at before trusting any
model's output on these frames.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import video

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"

CLIPS = [
    "11_john_marston_seg_0313",
    "26_trevor_seg_0004",
    "46_franklin_seg_0093",
    "17_michael_seg_0042",
    "48_franklin_seg_0100",
]
FRAME = 40


def saturation_and_edges(frame: np.ndarray) -> tuple[float, float]:
    """Cheap stand-ins for 'has colour' and 'has texture'."""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return float(hsv[..., 1].mean()), float((edges > 0).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(CLIPS), figsize=(4 * len(CLIPS), 4.6))

    for column, stem in enumerate(CLIPS):
        for row, kind in enumerate(("high", "low")):
            frames = video.read_frames(DATA / kind / f"{stem}.mp4", limit=FRAME + 1)
            frame = frames[FRAME]
            saturation, edge_density = saturation_and_edges(frame)
            axes[row, column].imshow(frame)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(
                f"{stem.split('_seg_')[0]}  [{kind}]\nsat {saturation:.0f}   edges {edge_density * 100:.1f}%",
                fontsize=9,
            )

    fig.suptitle(
        "High render vs low-poly render, same frame.\n"
        "The low side keeps layout and silhouettes but loses most colour and nearly all texture.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "appearance_gap.png", dpi=130)
    print(f"wrote {OUT / 'appearance_gap.png'}")


if __name__ == "__main__":
    main()
