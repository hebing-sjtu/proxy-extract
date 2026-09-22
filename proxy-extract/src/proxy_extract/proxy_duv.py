"""The `proxy_duv` deliverable: per-frame depth and semantics, in CWM class
space, laid out the way `PROXY_DUV_SPEC.md` asks for it.

    <root>/
        encode_manifest.jsonl     one JSON object per line, paths relative to root
        <segment>/
            video.mp4             the target; or target/rgb.mp4, from the clip route
            duv/
                000000.depth.f32        258048 bytes, little-endian float32, metres
                000000.semantic_id.png  L mode, 336x192, ids in [0, 12)
                ... through 000123
        ...

Section 7 puts every path in the manifest rather than fixing a tree, so the
segment directories are whatever produced them - `clips.py` writes `clip_*`
with the target under `target/rgb.mp4` - and only `duv/` is a fixed name,
because that is the one the consumer opens by ordinal.

This is the *first* of the two forms the consumer accepts, and the spec is
emphatic about why it is worth preferring: handing over a composed `duv.mp4`
means re-implementing the red channel's log curve, the twelve G/B codes and the
sky convention on this side, and the consumer validates none of it because none
of it is checkable - a depth map written over the wrong range still looks
exactly like a depth map. Per frame, there is no palette to get wrong.

Three things here are not obvious and all three are silent when wrong.

**`duv/` is a condition_root.** The file names, byte counts and PNG mode the
spec asks for are the ones `contract.py` already writes, so this module does not
re-implement them - it places one per segment and adds the statistics the
contract does not check. The two must not drift apart, and the way they are
kept together is that there is only one writer.

**The class ids are CWM's twelve, not the delivered eleven.** `delivery.py` and
`clips.py` write DATA_F.md's 11-class schema, and those ids are *valid* here -
every one of them is under 12, so nothing raises, no shape changes and the loss
curve looks normal. What changes is which class each id means. The projection is
`taxonomy.STANDARD11_TO_CWM`, transcribed from the spec, and it runs on the way
in rather than being left to the caller.

**The acceptance criterion is the depth percentiles, not the absence of
errors.** Spec section 8 spells this out and it is the only check here that can
catch the one fatal mistake: per-segment depth normalisation produces files that
pass every structural test, and the way it shows up is that p50 differs by
orders of magnitude between segments that were shot in similar places. So
`audit_root` compares segments against each other rather than each against a
threshold, and that comparison is the reason this is a corpus-level function
and not a per-segment one.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contract
from .taxonomy import CLASS_NAMES, NUM_CLASSES, to_cwm12

SEG_PREFIX = "seg_"
DUV_DIRNAME = "duv"
TARGET_NAME = "video.mp4"
ANCHOR_NAME = "anchor.png"
MANIFEST_NAME = "encode_manifest.jsonl"
SEMANTIC_NAME = "semantic.json"

# FastVideo's authoritative 4x3 semantic grid. U changes fastest: ids 0..3
# walk across the first row, then ids 4..7 the second. Swapping the axes still
# yields twelve distinct codes and therefore fails silently, so keep the
# formula in the metadata as well as its evaluated values.
SEMANTIC_U = (32, 96, 160, 224)
SEMANTIC_V = (43, 128, 213)

# The consumer's window. `--num-frames` must satisfy n % 17 == 5 for the H3
# causal VAE, and 124 is the default it ships with; a segment shorter than this
# is rejected outright rather than padded.
SPEC_FRAMES = contract.WINDOW_FRAMES

# The consumer's timeline is fixed at this, and the per-frame path does not
# resample - only the video path does. So a segment delivered this way has to
# be 24 fps already, and this constant exists to be quoted at a caller that
# passes something else rather than to be used in arithmetic.
SPEC_FPS = 24.0

# Percentiles the spec asks for. p50 is the one that matters across segments;
# p1 and p99 say whether the encoder's [0.3, 256] range is the right one.
SPEC_PERCENTILES = (1.0, 50.0, 99.0)

# How far the per-segment median depth may spread across a corpus before the
# audit calls it out. Expressed as a ratio of the 90th to the 10th percentile of
# the per-segment medians, so it asks "do these segments describe the same
# world", not "is any one of them unusual".
#
# Ten is deliberately loose. A corpus with real indoor and real aerial footage
# in it genuinely spans this, so tripping the check is a prompt to look rather
# than a verdict. What it exists to catch is the order-of-magnitude spread that
# per-segment normalisation produces, and that is not a near miss.
MAX_MEDIAN_SPREAD = 10.0


class ProxyDuvError(ValueError):
    """A delivery does not satisfy what PROXY_DUV_SPEC.md asks for."""


def seg_dir_for(root: Path, index: int) -> Path:
    return Path(root) / f"{SEG_PREFIX}{index:06d}"


def duv_dir_for(seg_dir: Path) -> Path:
    return Path(seg_dir) / DUV_DIRNAME


def semantic_uv_metadata() -> dict:
    """The CWM class-id to packed-DUV `(U, V)` mapping.

    Byte values are what an RGB DUV frame carries in G and B. FastVideo divides
    them by 255 before feeding the VAE; recording bytes rather than rounded
    floats keeps the mapping bit-exact and lets any consumer choose its own
    numeric representation.
    """
    width = len(SEMANTIC_U)
    return {
        "schema": "cwm12-semantic-uv",
        "version": 1,
        "resolution": {
            "width": contract.CONDITION_WIDTH,
            "height": contract.CONDITION_HEIGHT,
        },
        "semantic_id": {
            "file_pattern": "%06d.semantic_id.png",
            "png_mode": "L",
            "valid_range": [0, NUM_CLASSES - 1],
        },
        "uv_encoding": {
            "u_rgb_channel": "G",
            "v_rgb_channel": "B",
            "u_levels": list(SEMANTIC_U),
            "v_levels": list(SEMANTIC_V),
            "u_formula": "u_levels[semantic_id % 4]",
            "v_formula": "v_levels[semantic_id // 4]",
            "vae_normalization": "byte / 255.0",
        },
        "classes": {
            str(class_id): {
                "name": name,
                "u": SEMANTIC_U[class_id % width],
                "v": SEMANTIC_V[class_id // width],
            }
            for class_id, name in enumerate(CLASS_NAMES)
        },
    }


def write_semantic_json(duv_dir: Path) -> Path:
    """Atomically write the semantic/UV sidecar required beside DUV frames."""
    duv_dir = Path(duv_dir)
    duv_dir.mkdir(parents=True, exist_ok=True)
    path = duv_dir / SEMANTIC_NAME
    payload = json.dumps(semantic_uv_metadata(), indent=2, ensure_ascii=False) + "\n"
    handle, scratch = tempfile.mkstemp(dir=duv_dir, prefix=f"{SEMANTIC_NAME}.", suffix=".tmp")
    tmp = Path(scratch)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_frame(
    seg_dir: Path,
    ordinal: int,
    depth_metres: np.ndarray,
    labels: np.ndarray,
    *,
    taxonomy: str = "cwm12",
) -> None:
    """Write one ordinal's pair into a segment's `duv/`.

    `taxonomy` says what `labels` already are. It is required to be stated
    rather than sniffed, because both schemas are small integers in the same
    range: `standard11` ids pass every check this format applies and mean
    different classes, which is precisely the failure the spec opens by warning
    about. There is no value of the data that distinguishes them, so the
    caller has to.
    """
    duv = duv_dir_for(seg_dir)
    if ordinal == 0 or not (duv / SEMANTIC_NAME).is_file():
        write_semantic_json(duv)
    contract.write_frame(
        duv, ordinal, depth_metres, project_labels(labels, taxonomy)
    )


def project_labels(labels: np.ndarray, taxonomy: str) -> np.ndarray:
    """Put a label map into CWM's twelve classes, from whichever schema it is in."""
    if taxonomy == "cwm12":
        ids = np.asarray(labels, dtype=np.uint8)
        top = int(ids.max(initial=0))
        if top >= NUM_CLASSES:
            raise ProxyDuvError(
                f"labels declared cwm12 contain id {top}, outside [0, {NUM_CLASSES})"
            )
        return ids
    if taxonomy == "standard11":
        return to_cwm12(labels)
    raise ProxyDuvError(
        f"cannot project {taxonomy!r} onto the CWM classes. The spec requires a "
        "single corpus-wide table; add one to taxonomy.py rather than mapping at "
        "the call site, so that the whole corpus shares it."
    )


# ------------------------------------------------------------------- auditing


@dataclass(frozen=True)
class SegStats:
    """What one segment's `duv/` says about itself, per spec section 8."""

    seg: str
    frames: int
    valid_fraction: float
    percentiles: dict[str, float]
    below_near_fraction: float
    above_far_fraction: float
    classes_present: list[int]

    def as_dict(self) -> dict:
        return {
            "seg": self.seg,
            "frames": self.frames,
            "valid_fraction": self.valid_fraction,
            "metres_p1": self.percentiles["p1"],
            "metres_p50": self.percentiles["p50"],
            "metres_p99": self.percentiles["p99"],
            "below_near_fraction": self.below_near_fraction,
            "above_far_fraction": self.above_far_fraction,
            "classes_present": self.classes_present,
            "class_names_present": [CLASS_NAMES[cls] for cls in self.classes_present],
        }


def audit_seg(seg_dir: Path, *, frames: int = SPEC_FRAMES) -> SegStats:
    """Re-read one segment's `duv/` and compute the spec's acceptance numbers.

    Every structural check is delegated to `contract.validate_condition_root`,
    which is the same code path the consumer's loader asserts on. What is added
    here is the distribution, because "it did not raise" is not the bar: a
    segment whose depth is uniformly 80 m, or whose sky was written as a surface
    rather than as zero, passes every assertion and is useless.
    """
    seg_dir = Path(seg_dir)
    duv = duv_dir_for(seg_dir)
    if not duv.is_dir():
        raise ProxyDuvError(f"{seg_dir} has no {DUV_DIRNAME}/ directory")

    # Not `expected_frames=frames`: the spec asks for *at least* 124, and a
    # longer segment is fine as long as the ordinals are contiguous from zero.
    structural = contract.validate_condition_root(duv)
    if structural["frames"] < frames:
        raise ProxyDuvError(
            f"{seg_dir.name} is short of the window: {structural['frames']} frames "
            f"where {frames} are needed. The consumer opens range({frames}) and a "
            "missing ordinal is an ENOENT partway through building the cache."
        )

    valid_total = 0
    pixels = 0
    below = 0
    above = 0
    # Depth is sampled rather than pooled whole: a segment is 124 frames of
    # 64,512 pixels, and holding all 8 million to take three percentiles costs
    # 32 MB per segment for an answer a fixed stride gives to the same two
    # decimal places. The stride is over pixels, not frames, so every frame
    # still contributes - a per-frame sample would miss a single bad frame.
    sampled: list[np.ndarray] = []
    for ordinal in range(structural["frames"]):
        depth, _ = contract.read_frame(duv, ordinal)
        flat = depth.ravel()
        valid = flat > contract.DEPTH_VALID_EPSILON_METRES
        pixels += flat.size
        valid_total += int(valid.sum())
        metres = flat[valid]
        below += int((metres < contract.DEPTH_NEAR_METRES).sum())
        above += int((metres > contract.DEPTH_FAR_METRES).sum())
        sampled.append(metres[::_PERCENTILE_STRIDE])

    pooled = np.concatenate(sampled) if sampled else np.zeros(0, dtype=np.float32)
    percentiles = (
        dict(
            zip(
                ("p1", "p50", "p99"),
                (round(float(v), 4) for v in np.percentile(pooled, SPEC_PERCENTILES)),
            )
        )
        if pooled.size
        else {"p1": 0.0, "p50": 0.0, "p99": 0.0}
    )

    return SegStats(
        seg=seg_dir.name,
        frames=structural["frames"],
        valid_fraction=round(valid_total / max(pixels, 1), 6),
        percentiles=percentiles,
        below_near_fraction=round(below / max(valid_total, 1), 6),
        above_far_fraction=round(above / max(valid_total, 1), 6),
        classes_present=structural["semantic_classes_present"],
    )


_PERCENTILE_STRIDE = 8


def audit_root(root: Path, *, frames: int = SPEC_FRAMES) -> dict:
    """Audit every segment under `root`, and compare them against each other.

    The cross-segment comparison is the point. Each of the warnings below
    describes a delivery that passes every per-frame assertion the consumer
    makes and still cannot train:

      per-segment depth normalisation   medians spread over orders of magnitude
      sky written as a surface          valid_fraction at or near 1.0
      depth never written               valid_fraction at or near 0.0
      the wrong class table             a class the corpus should have, absent
    """
    root = Path(root)
    segs = segment_dirs(root)
    if not segs:
        raise ProxyDuvError(f"no segment holds a {DUV_DIRNAME}/ directory under {root}")

    stats: list[SegStats] = []
    failures: list[dict] = []
    for seg in segs:
        try:
            stats.append(audit_seg(seg, frames=frames))
        except (ValueError, OSError) as error:
            # ValueError rather than the two named subclasses: `contract` raises
            # bare ValueErrors from its byte-count and PNG-mode assertions, and
            # those are the most likely failures of all - a write cut short by
            # a full disk. Catching only the named ones let one bad segment end
            # the audit of two thousand, which is the opposite of the point.
            failures.append({"seg": seg.name, "error": f"{type(error).__name__}: {error}"})

    warnings: list[str] = []
    medians = sorted(item.percentiles["p50"] for item in stats if item.percentiles["p50"] > 0)
    spread = None
    if len(medians) >= 2:
        low, high = np.percentile(medians, [10.0, 90.0])
        spread = round(float(high / max(low, 1e-9)), 3)
        if spread > MAX_MEDIAN_SPREAD:
            warnings.append(
                f"per-segment median depth spreads {spread:.1f}x across the corpus "
                f"(p10 {low:.2f} m, p90 {high:.2f} m). Spec section 2: this is what "
                "per-frame or per-segment min/max normalisation looks like, and if "
                "that is what happened the whole batch is scrap. Check that the depth "
                "backend reported metric scale and that nothing renormalised after it."
            )

    for item in stats:
        if item.valid_fraction >= 0.999:
            warnings.append(
                f"{item.seg}: {item.valid_fraction:.1%} of pixels are valid, so the "
                "sky was written as a surface rather than as 0. A false far surface is "
                "ignored downstream but a ceiling at a finite depth is not."
            )
        elif item.valid_fraction <= 0.01:
            warnings.append(
                f"{item.seg}: only {item.valid_fraction:.2%} of pixels carry depth; "
                "the stack is effectively empty"
            )

    union: set[int] = set()
    for item in stats:
        union.update(item.classes_present)
    for item in stats:
        # Only for the classes the corpus does use. A corpus with no animals in
        # it is not a broken corpus, but one segment missing `sky` while every
        # other segment has it usually means that segment's labels came from a
        # different table.
        if 1 in union and 1 not in item.classes_present:
            warnings.append(f"{item.seg}: no `sky` pixels, unlike the rest of the corpus")

    return {
        "root": str(root),
        "segments": len(segs),
        "audited": len(stats),
        "failed": len(failures),
        "failures": failures[:20],
        "frames_each_at_least": frames,
        "median_spread": spread,
        "median_metres_p10_p90": (
            [round(float(v), 4) for v in np.percentile(medians, [10.0, 90.0])]
            if len(medians) >= 2
            else None
        ),
        "classes_present": sorted(union),
        "class_names_present": [CLASS_NAMES[cls] for cls in sorted(union)],
        "warnings": warnings,
        "segment_stats": [item.as_dict() for item in stats],
    }


# ------------------------------------------------------------------- manifest



# Where a segment's target video might be. The spec's own sketch says
# `<seg>/video.mp4`, and `clips.py` writes `<clip>/target/rgb.mp4`; both are
# legal, because section 7 puts every path in the manifest rather than fixing a
# tree. Probed in this order rather than configured, so that a corpus cut by
# either route packages without being told which one it was.
TARGET_CANDIDATES = (TARGET_NAME, "target/rgb.mp4")


def find_target(seg_dir: Path) -> Path | None:
    """The segment's target video, whichever layout produced it."""
    seg_dir = Path(seg_dir)
    for candidate in TARGET_CANDIDATES:
        path = seg_dir / candidate
        if path.is_file() and path.stat().st_size:
            return path
    return None


def manifest_entry(seg_dir: Path, root: Path, *, prompt: str | None = None) -> dict:
    """One manifest line for one segment, with every path relative to `root`.

    `proxy_duv` rather than `proxy_duv_video`, and never both: section 7 makes
    the three proxy keys mutually exclusive, so a segment that also happens to
    carry a composed `proxy/duv.mp4` - which is what the clip route writes
    alongside - must not mention it. That file is DATA_F.md's palette, not
    CWM's, and naming it here is the one way to get the consumer to read the
    wrong one of two files that are both present and both valid.

    `anchor` is omitted rather than pointed at frame 0. The spec is explicit
    that omitting it makes the consumer fall back to the target's own first
    frame, which is what is wanted, while supplying a nearly-identical image
    pins the appearance dictionary somewhere slightly wrong.
    """
    seg_dir, root = Path(seg_dir), Path(root)
    target = find_target(seg_dir)
    if target is None:
        raise ProxyDuvError(
            f"{seg_dir} has no target video; looked for {', '.join(TARGET_CANDIDATES)}"
        )
    name = seg_dir.name
    entry = {
        "name": name,
        "id": name,
        "target": str(target.relative_to(root)),
        "proxy_duv": str(duv_dir_for(seg_dir).relative_to(root)),
    }
    if prompt is not None:
        entry["prompt"] = prompt
    return entry


def write_manifest(
    root: Path, entries: list[dict], *, name: str = MANIFEST_NAME
) -> Path:
    """Write the manifest as JSON Lines, atomically.

    One object per line rather than one array, because the consumer reads it a
    line at a time and because a corpus manifest is appended to across runs. A
    truncated array is unparseable; a truncated JSONL file is short by one
    segment, which the audit will say.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)

    handle, scratch = tempfile.mkstemp(dir=root, prefix=f"{name}.", suffix=".tmp")
    tmp = Path(scratch)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(payload)
        os.replace(tmp, root / name)
    finally:
        tmp.unlink(missing_ok=True)
    return root / name


def segment_dirs(root: Path) -> list[Path]:
    """Directories under `root` that hold a `duv/`, in name order.

    Any directory, not just `seg_*`: the clip route names its output
    `clip_000123_2`, and the spec cares what the manifest says a segment is
    called, not what its directory is prefixed with.
    """
    root = Path(root)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and duv_dir_for(path).is_dir()
    )


# What `clip-prompts captions-export --write-txt` leaves in each segment: the
# flat CWM user sentence, already canonicalised. See CWM_TEXT_EXPORT.md.
PROMPT_TXT_NAME = "prompt.txt"


def prompt_beside(seg_dir: Path) -> str | None:
    """The exported user sentence sitting in a segment, if there is one.

    Read as bytes and decoded rather than through `read_text`, which applies
    universal newline translation and would turn the CRLF the export
    deliberately wrote into LF. CWM's released Qwen caches were encoded from
    CRLF bytes, so that rewrite changes the token ids of every caption while
    leaving a file that looks identical in an editor.
    """
    path = Path(seg_dir) / PROMPT_TXT_NAME
    if not path.is_file():
        return None
    text = path.read_bytes().decode("utf-8")
    return text or None


def manifest_from_root(
    root: Path, *, prompts: dict[str, str] | None = None
) -> list[dict]:
    """Scan `root` for finished segments and build the manifest from the tree.

    Only segments that have both a target and a `duv/` are listed: a half-cut
    segment in a manifest is a training run that dies partway through its first
    epoch, hours after the run that produced it finished.

    A segment's own `prompt.txt` is used when `prompts` does not name it, so the
    file `clip-prompts` was told to write is the file this reads. An explicit
    mapping still wins, because that is the only way to caption a corpus whose
    text came from somewhere else.
    """
    root = Path(root)
    prompts = prompts or {}
    entries = []
    for seg in segment_dirs(root):
        if find_target(seg) is None:
            continue
        prompt = prompts.get(seg.name)
        if prompt is None:
            prompt = prompt_beside(seg)
        entries.append(manifest_entry(seg, root, prompt=prompt))
    return entries


# The H3 causal VAE's constraint on a sample's length. This is *not*
# contract.py's 124 + 90k: that stride is code-world-model's windowing, and 214
# satisfies it while failing this. The two deliverables share their first window
# and nothing after it, which is why they get separate arithmetic rather than
# one helper that tries to serve both.
VAE_FRAME_MODULUS = 17
VAE_FRAME_REMAINDER = 5


def frames_are_deliverable(count: int) -> bool:
    """Whether a frame count satisfies the consumer's VAE window constraint."""
    return count >= SPEC_FRAMES and count % VAE_FRAME_MODULUS == VAE_FRAME_REMAINDER


def largest_deliverable_frame_count(count: int) -> int:
    """The longest deliverable segment that fits in `count` decoded frames."""
    if count < SPEC_FRAMES:
        raise ProxyDuvError(
            f"{count} frames is short of the {SPEC_FRAMES}-frame window; the "
            "consumer rejects a short segment rather than padding it"
        )
    return count - (count - VAE_FRAME_REMAINDER) % VAE_FRAME_MODULUS


def seconds_for(count: int, fps: float = SPEC_FPS) -> float:
    return math.floor(count / fps * 100) / 100
