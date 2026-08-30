"""Render a written condition_root back to something a human can judge.

The on-disk form is raw float32 and an 8-bit PNG whose values all look black, so
without this the only way to review a run is to train on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import CONDITION_HEIGHT, CONDITION_WIDTH, encode_depth_codes, read_frame
from .taxonomy import CLASS_NAMES, COARSE6_NAMES

# Distinguishable at a glance and roughly mnemonic: sky blue, water deep blue,
# terrain brown, road grey, vegetation green, human red.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (135, 206, 235),
    2: (30, 80, 200),
    3: (150, 110, 70),
    4: (110, 110, 110),
    5: (60, 160, 60),
    6: (210, 180, 140),
    7: (255, 165, 0),
    8: (230, 40, 40),
    9: (255, 220, 60),
    10: (170, 60, 200),
    11: (0, 190, 190),
}

COARSE6_COLORS: dict[int, tuple[int, int, int]] = {
    0: (70, 70, 78),
    1: (135, 135, 135),
    2: (60, 160, 60),
    3: (170, 60, 200),
    4: (255, 190, 40),
    5: (230, 40, 40),
}


@dataclass(frozen=True)
class Palette:
    """Class names and colours for one taxonomy.

    Both taxonomies write the same file format with small integer IDs, so a
    condition_root gives no clue which one produced it. Rendering 6-class
    output through the 12-class palette would paint `road` as `road_paved` and
    `hero` as `vegetation` — plausible-looking and wrong, which is worse than
    an error. Hence the explicit choice, resolved from the run's own report.
    """

    name: str
    names: tuple[str, ...]
    colors: dict[int, tuple[int, int, int]]

    @property
    def size(self) -> int:
        return len(self.names)

    def table(self) -> np.ndarray:
        palette = np.zeros((self.size, 3), dtype=np.uint8)
        for cls, color in self.colors.items():
            palette[cls] = color
        return palette


CWM12 = Palette("cwm12", CLASS_NAMES, CLASS_COLORS)
COARSE6 = Palette("coarse6", COARSE6_NAMES, COARSE6_COLORS)
PALETTES = {palette.name: palette for palette in (CWM12, COARSE6)}


def palette_for(condition_root: Path) -> Palette:
    """Which taxonomy a written condition_root used.

    Read from `extraction_report.json` rather than guessed from the highest ID
    present: a 12-class clip that happens to contain only sky and road would
    look exactly like a 6-class one.
    """
    report_path = Path(condition_root) / "extraction_report.json"
    if not report_path.exists():
        return CWM12
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return CWM12
    return PALETTES.get(report.get("semantic", {}).get("taxonomy", "cwm12"), CWM12)


def colorize_semantic(labels: np.ndarray, palette: Palette = CWM12) -> np.ndarray:
    table = palette.table()
    return table[np.clip(labels, 0, palette.size - 1)]


def colorize_depth(depth_metres: np.ndarray) -> np.ndarray:
    """Shade using the same log encoding the model will see, not linear metres.

    A linear ramp over 0.3-256 m renders almost everything black and hides
    exactly the near-field detail that matters.
    """
    import cv2

    codes = encode_depth_codes(depth_metres)
    gray = (codes.astype(np.float32) / 257.0).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[codes == 0] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _legend(width: int, height: int, present: list[int], palette: Palette) -> np.ndarray:
    import cv2

    strip = np.zeros((height, width, 3), dtype=np.uint8)
    if not present:
        return strip
    step = width // len(present)
    for slot, cls in enumerate(present):
        x0 = slot * step
        cv2.rectangle(strip, (x0, 0), (x0 + step - 2, height), palette.colors[cls][::-1], -1)
        cv2.putText(
            strip, palette.names[cls][:12], (x0 + 4, height - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)


def render_preview(
    condition_root: Path,
    out_path: Path,
    *,
    fps: int = 24,
    scale: int = 2,
    palette: Palette | None = None,
) -> Path:
    """Write an MP4 with depth beside semantics, plus a class legend."""
    import cv2

    condition_root, out_path = Path(condition_root), Path(out_path)
    palette = palette or palette_for(condition_root)
    ordinals = sorted(int(p.name[:6]) for p in condition_root.glob("??????.depth.f32"))
    if not ordinals:
        raise FileNotFoundError(f"no condition frames in {condition_root}")

    width, height = CONDITION_WIDTH * scale, CONDITION_HEIGHT * scale
    legend_height = 22
    out_path.parent.mkdir(parents=True, exist_ok=True)

    present: set[int] = set()
    panels: list[np.ndarray] = []
    for ordinal in ordinals:
        depth, semantic = read_frame(condition_root, ordinal)
        present.update(np.unique(semantic).tolist())
        pair = np.hstack([colorize_depth(depth), colorize_semantic(semantic, palette)])
        panels.append(cv2.resize(pair, (width * 2, height), interpolation=cv2.INTER_NEAREST))

    strip = _legend(width * 2, legend_height, sorted(present), palette)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height + legend_height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a writer for {out_path}")
    try:
        for panel in panels:
            frame = np.vstack([panel, strip])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return out_path
