"""End-to-end extraction: video in, a code-world-model `condition_root` out."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import contract
from .cameras import CameraTrack
from .depth import get_backend as get_depth_backend
from .depth.scale import apply_range_guard, solve_scale_from_cameras
from .semantic import get_backend as get_semantic_backend
from .semantic import get_refiner
from .taxonomy import CLASS_NAMES, COARSE6_NAMES
from .temporal import flicker_rate, stabilize_depth, stabilize_labels
from .video import read_frames

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
    # Only meaningful with the coarse6 backend: nothing predicts `hero`, so it
    # is derived from tracking after the labels exist.
    split_hero: bool = True

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

    frames = read_frames(video_path, size=config.work_size)
    keep = config.usable_frame_count(len(frames))
    frames = frames[:keep]
    if cameras is not None:
        cameras = cameras.subset(np.arange(keep))

    depth_backend = depth_backend or get_depth_backend(
        config.depth_backend, **config.depth_backend_options
    )
    semantic_backend = semantic_backend or get_semantic_backend(
        config.semantic_backend, **config.semantic_backend_options
    )
    if refiner is None:
        refiner = get_refiner(config.semantic_refiner, **config.refiner_options)

    depth_result = depth_backend.estimate(frames, cameras=cameras)
    metric_depth, calibration = _calibrate(depth_result, cameras)

    semantic_result = semantic_backend.segment(frames)
    if refiner is not None:
        semantic_result = refiner.refine(frames, semantic_result)

    # Reduce to the condition grid before stabilising: the temporal passes are
    # the expensive part and gain nothing from running at 16x the pixel count.
    small_depth = np.stack(
        [contract.downsample_depth(d, mode=config.depth_downsample) for d in metric_depth]
    )
    small_labels = np.stack([contract.downsample_semantic(l) for l in semantic_result.labels])

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

    # After stabilising, so the tracker sees settled masks rather than chasing
    # per-frame flicker into spurious tracks.
    hero_info: dict = {"attempted": False}
    if config.split_hero and config.semantic_backend == "coarse6":
        from .semantic.hero import split as split_hero

        result = split_hero(stable_labels)
        stable_labels = result.labels
        hero_info = {
            "attempted": True,
            "resolved": result.hero_track is not None,
            "note": result.note,
            "person_tracks": len(result.tracks),
            "multi_person_frames": round(result.multi_person_frames, 4),
            "merged_frames": round(result.merged_frames, 4),
        }

    guarded, range_stats = apply_range_guard(
        stable_depth, near=contract.DEPTH_NEAR_METRES, far=contract.DEPTH_FAR_METRES
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for ordinal in range(keep):
        contract.write_frame(output_root, ordinal, guarded[ordinal], stable_labels[ordinal])

    validation = contract.validate_condition_root(output_root, expected_frames=keep)
    names = COARSE6_NAMES if config.semantic_backend == "coarse6" else CLASS_NAMES
    histogram = {
        names[cls]: round(float((stable_labels == cls).mean()), 6) for cls in np.unique(stable_labels)
    }

    report = {
        "clip": video_path.stem,
        "source_video": str(video_path),
        "condition_root": str(output_root),
        "frames": keep,
        "decoded_frames": len(frames),
        "config": {**asdict(config), "work_size": list(config.work_size)},
        "depth": {
            "backend": getattr(depth_backend, "name", config.depth_backend),
            **calibration,
            **range_stats,
            "meta": depth_result.meta,
        },
        "semantic": {
            "backend": getattr(semantic_backend, "name", config.semantic_backend),
            "refiner": getattr(refiner, "name", None),
            "class_fractions": histogram,
            "taxonomy": "coarse6" if config.semantic_backend == "coarse6" else "cwm12",
            "hero_split": hero_info,
            "flicker_before": flicker_before,
            "flicker_after": flicker_rate(stable_labels),
            "meta": semantic_result.meta,
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


def shard(clips: list[Path], index: int, count: int) -> list[Path]:
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
