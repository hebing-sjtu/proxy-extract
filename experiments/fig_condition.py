"""Figure: the artefact itself, read back off disk.

Every other figure here argues about whether a step works. This one just shows
what a finished clip looks like after `proxy-extract extract` — source frame,
the depth CWM will see (log-encoded, not linear metres), and the 6-class map —
so a newcomer following the runbook has something to compare their own run
against.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import contract, preview, taxonomy, video

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
COND = ROOT / "experiments" / "cond_c6" / "high"
OUT = ROOT / "experiments" / "figures"

CLIP = "26_trevor_seg_0004"
FRAMES = [0, 30, 60, 90]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = COND / CLIP
    report = json.loads((root / "extraction_report.json").read_text())
    palette = preview.palette_for(root)

    rgb = video.read_frames(
        DATA / "high" / f"{CLIP}.mp4", size=(contract.CONDITION_WIDTH, contract.CONDITION_HEIGHT)
    )

    # DUV codes run 0.3 m -> 65535 and 256 m -> 0, so the turbo ramp reads warm
    # for near. Worth spelling out: everyone's first instinct is the opposite.
    rows = (
        "source frame\n(1344x768 -> 336x192)",
        "depth\n(log DUV codes; warm = near)",
        f"semantics\n({palette.name})",
    )
    fig, axes = plt.subplots(3, len(FRAMES), figsize=(3.5 * len(FRAMES), 6.8), constrained_layout=True)

    for column, index in enumerate(FRAMES):
        depth, labels = contract.read_frame(root, index)
        axes[0, column].imshow(rgb[index])
        axes[0, column].set_title(f"frame {index}", fontsize=9)
        axes[1, column].imshow(preview.colorize_depth(depth))
        finite = depth[np.isfinite(depth) & (depth > 0)]
        axes[1, column].set_title(
            f"{finite.min():.1f}-{np.percentile(finite, 99):.0f} m", fontsize=9
        )
        axes[2, column].imshow(preview.colorize_semantic(labels, palette))
        present = sorted(np.unique(labels).tolist())
        axes[2, column].set_title(
            " ".join(palette.names[c] for c in present), fontsize=7.5
        )

    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        mpatches.Patch(color=np.array(palette.colors[c]) / 255, label=palette.names[c])
        for c in range(palette.size)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=palette.size,
        fontsize=9,
        frameon=False,
    )

    validation = report["validation"]
    fig.suptitle(
        f"{CLIP}: the written condition_root, decoded with CWM's own reader\n"
        f"{validation['frames']} frames  |  depth {report['depth']['backend']}  |  "
        f"semantics {report['semantic']['backend']}  |  "
        f"flicker {report['semantic']['flicker_before']:.3f} -> "
        f"{report['semantic']['flicker_after']:.3f}",
        fontsize=11,
    )
    fig.savefig(OUT / "condition_output.png", dpi=135, bbox_inches="tight")
    print(f"wrote {OUT / 'condition_output.png'}")


if __name__ == "__main__":
    main()
