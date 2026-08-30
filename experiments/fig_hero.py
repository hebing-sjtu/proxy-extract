"""Figure: did the tracker actually pick the protagonist?

Runs on the one clip in the sample that has other people in frame for a
meaningful share of its length, since that is the only place the decision is
exercised.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import contract, preview, taxonomy, video

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"

# Shared with the preview renderer so a figure and its MP4 never disagree.
COARSE6_COLORS = preview.COARSE6.colors

CLIP = "11_john_marston_seg_0313"
FRAMES = [8, 40, 72, 104]


def colorize(labels: np.ndarray) -> np.ndarray:
    return preview.colorize_semantic(labels, preview.COARSE6)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = ROOT / "experiments" / "cond_c6" / "high" / CLIP
    rgb = video.read_frames(
        DATA / "high" / f"{CLIP}.mp4", size=(contract.CONDITION_WIDTH, contract.CONDITION_HEIGHT)
    )

    fig, axes = plt.subplots(2, len(FRAMES), figsize=(3.6 * len(FRAMES), 4.4), constrained_layout=True)
    for column, index in enumerate(FRAMES):
        _, labels = contract.read_frame(root, index)
        axes[0, column].imshow(rgb[index])
        axes[0, column].set_title(f"frame {index}", fontsize=9)
        axes[1, column].imshow(colorize(labels))
        people = {
            taxonomy.COARSE6_NAMES[c]: int(np.sum(labels == c))
            for c in (taxonomy.C6_HERO, taxonomy.C6_NPC)
        }
        axes[1, column].set_title(
            f"hero {people['hero']} px   npc {people['npc']} px", fontsize=9
        )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        mpatches.Patch(color=np.array(COARSE6_COLORS[c]) / 255, label=taxonomy.COARSE6_NAMES[c])
        for c in range(taxonomy.NUM_COARSE6)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(
        f"{CLIP}: 6-class condition read back off disk, hero picked by tracking\n"
        "the only sample clip where other people are on screen (44% of frames)",
        fontsize=11,
    )
    fig.savefig(OUT / "hero_split.png", dpi=135, bbox_inches="tight")
    print(f"wrote {OUT / 'hero_split.png'}")


if __name__ == "__main__":
    main()
