"""End-to-end extraction: video in, a code-world-model `condition_root` out."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np

from . import contract
from .cameras import CameraTrack
from .depth import get_backend as get_depth_backend
from .depth.base import DepthResult
from .depth.scale import apply_range_guard, solve_scale_from_cameras
from .semantic import get_backend as get_semantic_backend
from .semantic import get_refiner
from .semantic.people import split_people
from .taxonomy import NAMES_OF_TAXONOMY
from .temporal import flicker_rate, stabilize_depth, stabilize_labels
from .video import iter_frames, read_frames

# Which label set a backend emits.
_TAXONOMY_OF_BACKEND = {"coarse6": "coarse6", "standard11": "standard11"}

# `shard` partitions clip paths for the condition pipeline and scene
# assignments for the delivery one; the striding is identical either way.
_T = TypeVar("_T")

# The resolution the heavy models see. 1344x768 matches the inference target and
# is an exact 4x multiple of the 336x192 condition grid, which lets the
# downsamplers use clean block reductions instead of resampling.
WORK_WIDTH = 1344
WORK_HEIGHT = 768


@dataclass
class ExtractionConfig:
    depth_backend: str = "mapanything"
    semantic_backend: str = "ade20k"
    semantic_refiner: str = "none"
    temporal_radius: int = 2
    flow_compensate: bool = True
    depth_downsample: str = "median"
    work_size: tuple[int, int] = (WORK_WIDTH, WORK_HEIGHT)
    depth_backend_options: dict = field(default_factory=dict)
    semantic_backend_options: dict = field(default_factory=dict)
    refiner_options: dict = field(default_factory=dict)
    # No segmenter predicts the protagonist, so it is derived from framing after
    # the labels exist. Applies to both person-splitting taxonomies: coarse6
    # calls the pair hero/npc, standard11 calls it player/ped.
    split_hero: bool = True
    # Frames per inference batch, or None to hold the whole clip at once. Only
    # worth setting for episodes far longer than the 124-frame window this was
    # built around; see `_reduce_in_chunks` for what it costs and buys.
    chunk_frames: int | None = None

    @property
    def taxonomy(self) -> str:
        return _TAXONOMY_OF_BACKEND.get(self.semantic_backend, "cwm12")

    def usable_frame_count(self, decoded: int) -> int:
        """Largest 124 + 90k frame count that fits in `decoded` frames."""
        windows = contract.window_count_for(decoded)
        if windows < 1:
            raise ValueError(
                f"clip has {decoded} frames but a window needs {contract.WINDOW_FRAMES}"
            )
        return contract.frames_for_windows(windows)


def _calibrate(depth_result, cameras: CameraTrack | None) -> tuple[np.ndarray, dict]:
    """Put depth into metres, preferring the GT camera baseline over the model."""
    info: dict = {"metric_source": None}

    if cameras is not None and depth_result.cam2world is not None:
        solution = solve_scale_from_cameras(depth_result.cam2world[:, :3, 3], cameras.positions)
        info["camera_scale_solution"] = asdict(solution)
        if solution.solved:
            info["metric_source"] = "gt_camera_baseline"
            return depth_result.scaled(solution.scale).depth, info

    if depth_result.metric:
        info["metric_source"] = "backend_native"
        return np.where(depth_result.valid_mask(), depth_result.depth, 0.0).astype(np.float32), info

    raise ValueError(
        "depth is up-to-scale and no usable GT camera track was available; "
        "supply --cameras or use a backend that predicts metric scale"
    )


@dataclass
class _Reduced:
    """One clip's models output, already on the 336x192 condition grid."""

    depth: np.ndarray  # (N, 192, 336) float32, backend units, invalid = 0
    labels: np.ndarray  # (N, 192, 336) uint8
    decoded: int
    metric: bool
    cam2world: np.ndarray | None
    depth_meta: dict
    semantic_meta: dict
    batches: int


def _reduce_in_chunks(
    video_path: Path,
    config: ExtractionConfig,
    *,
    depth_backend,
    semantic_backend,
    refiner,
) -> _Reduced:
    """Run the heavy models in batches, keeping only the reduced result.

    A long episode's full-resolution intermediates do not fit in memory
    together: 1744 frames at the 1344x768 work size is 5.4 GB of RGB alone, and
    the float32 depth stack derived from it another 7.2 GB. Reducing each batch
    to the condition grid and dropping it bounds peak usage by the batch instead
    of the episode, at 112 MB per accumulated stack.

    Reducing before calibrating is exact rather than a convenient
    approximation: the block reducers are medians, minima and means, and all
    three commute with the positive scalar that calibration multiplies through.

    What it does cost is any cross-batch reasoning inside the depth backend.
    Backends are handed whole clips precisely so the multi-view ones can derive
    temporal consistency from seeing the sequence jointly, and a batched call
    throws that away at every seam. Per-frame backends are unaffected.
    """
    depth_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    cam_parts: list[np.ndarray] = []
    depth_meta: dict = {}
    semantic_meta: dict = {}
    metric = True
    decoded = 0
    batches = 0

    stream = iter_frames(video_path, size=config.work_size, chunk=config.chunk_frames or 1 << 30)
    for batch in stream:
        depth_result = depth_backend.estimate(batch, cameras=None)
        semantic_result = semantic_backend.segment(batch)
        if refiner is not None:
            semantic_result = refiner.refine(batch, semantic_result)

        # Zero the invalid pixels before reducing so that a block which is
        # entirely invalid stays invalid rather than borrowing a neighbour.
        raw = np.where(depth_result.valid_mask(), depth_result.depth, 0.0).astype(np.float32)
        depth_parts.append(
            np.stack([contract.downsample_depth(d, mode=config.depth_downsample) for d in raw])
        )
        label_parts.append(
            np.stack([contract.downsample_semantic(l) for l in semantic_result.labels])
        )
        if depth_result.cam2world is not None:
            cam_parts.append(np.asarray(depth_result.cam2world))

        metric = metric and depth_result.metric
        depth_meta, semantic_meta = depth_result.meta, semantic_result.meta
        decoded += len(batch)
        batches += 1

    if not depth_parts:
        raise ValueError(f"decoded zero frames from {video_path}")

    return _Reduced(
        depth=np.concatenate(depth_parts),
        labels=np.concatenate(label_parts),
        decoded=decoded,
        metric=metric,
        cam2world=np.concatenate(cam_parts) if len(cam_parts) == batches else None,
        depth_meta=depth_meta,
        semantic_meta=semantic_meta,
        batches=batches,
    )


def extract_clip(
    video_path: Path,
    output_root: Path,
    *,
    config: ExtractionConfig | None = None,
    cameras: CameraTrack | None = None,
    depth_backend=None,
    semantic_backend=None,
    refiner=None,
) -> dict:
    """Extract one clip's depth + semantic condition into `output_root`.

    Backends may be passed in already constructed so that a batch run loads
    each model once rather than once per clip.
    """
    config = config or ExtractionConfig()
    video_path, output_root = Path(video_path), Path(output_root)
    started = time.time()

    if config.chunk_frames is not None and cameras is not None:
        raise ValueError(
            "--chunk-frames cannot be combined with a GT camera track: each batch is "
            "an independent reconstruction, so the predicted poses do not share a "
            "coordinate frame and a single scale cannot be solved across them. "
            "Drop one of the two."
        )

    depth_backend = depth_backend or get_depth_backend(
        config.depth_backend, **config.depth_backend_options
    )
    semantic_backend = semantic_backend or get_semantic_backend(
        config.semantic_backend, **config.semantic_backend_options
    )
    if refiner is None:
        refiner = get_refiner(config.semantic_refiner, **config.refiner_options)

    reduced = _reduce_in_chunks(
        video_path,
        config,
        depth_backend=depth_backend,
        semantic_backend=semantic_backend,
        refiner=refiner,
    )
    keep = config.usable_frame_count(reduced.decoded)
    if cameras is not None:
        cameras = cameras.subset(np.arange(keep))

    # Calibration runs on the reduced stack. `_calibrate` only needs the depth
    # values, the metric flag and the poses, all of which survive reduction.
    calibrated, calibration = _calibrate(
        DepthResult(
            depth=reduced.depth[:keep],
            metric=reduced.metric,
            cam2world=None if reduced.cam2world is None else reduced.cam2world[:keep],
            meta=reduced.depth_meta,
        ),
        cameras,
    )
    small_depth = calibrated
    small_labels = reduced.labels[:keep]

    guide = None
    if config.flow_compensate:
        guide = read_frames(
            video_path,
            size=(contract.CONDITION_WIDTH, contract.CONDITION_HEIGHT),
            limit=keep,
            grayscale=True,
        )

    flicker_before = flicker_rate(small_labels)
    stable_depth = stabilize_depth(small_depth, guide_frames=guide, radius=config.temporal_radius)
    stable_labels = stabilize_labels(small_labels, guide_frames=guide, radius=config.temporal_radius)

    stable_labels, hero_info = split_people(
        stable_labels, taxonomy=config.taxonomy, enabled=config.split_hero
    )

    guarded, range_stats = apply_range_guard(
        stable_depth, near=contract.DEPTH_NEAR_METRES, far=contract.DEPTH_FAR_METRES
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for ordinal in range(keep):
        contract.write_frame(output_root, ordinal, guarded[ordinal], stable_labels[ordinal])

    validation = contract.validate_condition_root(output_root, expected_frames=keep)
    names = NAMES_OF_TAXONOMY[config.taxonomy]
    histogram = {
        names[cls]: round(float((stable_labels == cls).mean()), 6) for cls in np.unique(stable_labels)
    }

    report = {
        "clip": video_path.stem,
        "source_video": str(video_path),
        "condition_root": str(output_root),
        "frames": keep,
        "decoded_frames": reduced.decoded,
        "inference_batches": reduced.batches,
        "config": {**asdict(config), "work_size": list(config.work_size)},
        "depth": {
            "backend": getattr(depth_backend, "name", config.depth_backend),
            **calibration,
            **range_stats,
            "meta": reduced.depth_meta,
        },
        "semantic": {
            "backend": getattr(semantic_backend, "name", config.semantic_backend),
            "refiner": getattr(refiner, "name", None),
            "class_fractions": histogram,
            "taxonomy": config.taxonomy,
            "hero_split": hero_info,
            "flicker_before": flicker_before,
            "flicker_after": flicker_rate(stable_labels),
            "meta": reduced.semantic_meta,
        },
        "validation": validation,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_root / "extraction_report.json").write_text(json.dumps(report, indent=2))
    return report


def condition_dir_for(output_root: Path, clip: Path) -> Path:
    """Where one clip's condition_root goes, under `output_root`.

    The source track is part of the path because the high and low renders of a
    pair share a file name. Keying on the stem alone would make the second
    extraction silently overwrite the first, which is exactly the "extract both
    tracks and compare" workflow this exists for.
    """
    clip = Path(clip)
    return Path(output_root) / clip.parent.name / clip.stem


def shard(clips: list[_T], index: int, count: int) -> list[_T]:
    """The slice of `clips` this worker owns, out of `count` workers.

    Strided rather than contiguous so that a run whose cost varies along the
    clip list - one character's scenes being heavier than another's - still
    spreads evenly, and so that adding a worker does not reshuffle everything.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for {count} workers")
    return [clip for position, clip in enumerate(clips) if position % count == index]


def already_done(output_root: Path, clip: Path, expected_frames: int | None = None) -> bool:
    """Whether a previous run left a complete, readable condition_root here.

    Checked by re-validating the files rather than by trusting a marker, so a
    worker killed mid-write leaves something that fails this test and gets
    redone instead of a half-clip that silently enters the dataset.
    """
    directory = condition_dir_for(output_root, clip)
    if not (directory / "extraction_report.json").exists():
        return False
    try:
        contract.validate_condition_root(directory, expected_frames=expected_frames)
    except (ValueError, OSError, KeyError):
        return False
    return True


def extract_dataset(
    clips: list[Path],
    output_root: Path,
    *,
    config: ExtractionConfig | None = None,
    cameras: dict[str, CameraTrack] | None = None,
    resume: bool = False,
    on_error: str = "raise",
) -> list[dict]:
    """Run `extract_clip` over many clips, loading each model exactly once."""
    if on_error not in {"raise", "skip"}:
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    config = config or ExtractionConfig()
    depth_backend = get_depth_backend(config.depth_backend, **config.depth_backend_options)
    semantic_backend = get_semantic_backend(config.semantic_backend, **config.semantic_backend_options)
    refiner = get_refiner(config.semantic_refiner, **config.refiner_options)

    reports = []
    for clip in clips:
        clip = Path(clip)
        if resume and already_done(output_root, clip):
            reports.append({"clip": clip.stem, "skipped": "already complete"})
            continue
        try:
            reports.append(
                extract_clip(
                    clip,
                    condition_dir_for(output_root, clip),
                    config=config,
                    cameras=(cameras or {}).get(clip.stem),
                    depth_backend=depth_backend,
                    semantic_backend=semantic_backend,
                    refiner=refiner,
                )
            )
        except Exception as error:
            # In a batch of thousands, one unreadable file should cost one clip,
            # not the worker's whole shard.
            if on_error == "raise":
                raise
            reports.append({"clip": clip.stem, "failed": f"{type(error).__name__}: {error}"})
    return reports
