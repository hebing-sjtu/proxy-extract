"""Predict a gta-web style delivery scene from a plain RGB episode.

`pipeline.extract_clip` writes the `condition_root` that code-world-model
consumes: raw float32 depth and 8-bit IDs on the 336x192 grid its windows are
cut to. This module writes the other deliverable, the one DATA_F.md specifies —
four aligned videos at 1280x720, full episode length, plus the source
annotations:

    seg_long_000000/
        color.mp4           RGB, libx264 / yuv420p
        depth.mp4           inverted log-z, 8-bit grey, lossless
        semantic.mp4        (R, G, B) = (0, 0, id), lossless RGB
        duv.mp4             R = log-z, G/B = semantic colour, lossless RGB
        annotation.tar      the episode's own annotations, verbatim
        extraction_report.json

The 336x192 reduction is deliberately *not* applied here. It throws away 93% of
the pixels and it is cheap to redo from these videos later, so doing it now
would only mean the choice of grid could never be revisited. Nothing else is
held back: every decoded frame is delivered, at the source frame rate.

1280x720 is exactly two-thirds of ABot's 1920x1080, so the resize introduces no
aspect distortion, and it is the resolution `semantic.player`'s priors were
fitted at on the real gta-web corpus — so the protagonist split runs on native
pixels here rather than on a proxy grid.

The colour video is encoded from the same decoded frames the models saw, rather
than by handing the source file to ffmpeg separately. Two resamplers do not
agree to the pixel, and a delivery set whose RGB is a fraction of a pixel off
from its own depth is worse than useless for anything that learns a
correspondence.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import proxy
from .depth import get_backend as get_depth_backend
from .depth.scale import apply_range_guard
from .semantic import get_backend as get_semantic_backend
from .semantic import get_refiner
from .semantic.people import split_people
from .taxonomy import NAMES_OF_TAXONOMY
from .temporal import flicker_rate, stabilize_depth, stabilize_labels
from .video import iter_frames, probe

DELIVERY_WIDTH = 1280
DELIVERY_HEIGHT = 720

# DATA_F.md's own depth video range, which the encoders in `proxy` implement.
# Wider on the near side than the condition contract's 0.3 m.
DELIVERY_NEAR_METRES = proxy.DEPTH_VIDEO_NEAR_METRES
DELIVERY_FAR_METRES = proxy.DEPTH_VIDEO_FAR_METRES

VIDEO_NAMES = {
    "color": "color.mp4",
    "depth": "depth.mp4",
    "semantic": "semantic.mp4",
    "duv": "duv.mp4",
}

# `proxy.open_encoder` keys its pixel formats off DATA_F.md's own stream names,
# where the packed depth-plus-semantics composition is called `proxy`.
#
# It is delivered as `duv.mp4` because the idea is code-world-model's DUV frame:
# depth in one channel, semantics in the other two. The two are NOT
# interchangeable, and assuming they are would corrupt a training run silently.
# CWM's DUV carries the depth log code over near 0.3 / far 256 in R and the
# semantic palette's (u, v) in G/B; this carries DATA_F.md's log-z over near 0.1
# / far 8000 in R, with 255 reserved for sky, and a semantic *colour* in G/B.
# `cwm_h3_inference.duv` reads a condition_root, not this file.
_ENCODER_KIND = {"color": "color", "depth": "depth", "semantic": "semantic", "duv": "proxy"}

ANNOTATION_NAME = "annotation.tar"
REPORT_NAME = "extraction_report.json"

# Long-form segments: one directory per episode, full length, no window cutting.
SCENE_PREFIX = "seg_long_"

_TAXONOMY_OF_BACKEND = {"coarse6": "coarse6", "standard11": "standard11"}

# Backends that fabricate their output. They exist so the pipeline can be
# exercised without a GPU, and their videos are structurally indistinguishable
# from real ones — same size, same codecs, same class ids — which is exactly why
# a run using them has to say so in the report rather than leaving a reader to
# infer it from a backend name.
PLACEHOLDER_BACKENDS = frozenset({"synthetic"})


class DeliveryError(RuntimeError):
    pass


class PlaceholderOutput(UserWarning):
    """Raised as a warning when a scene is built from fabricated predictions."""


def placeholder_backends(depth_backend, semantic_backend, config: DeliveryConfig) -> list[str]:
    """Names of the placeholder backends this run used, in report order."""
    used = [
        ("depth", getattr(depth_backend, "name", config.depth_backend)),
        ("semantic", getattr(semantic_backend, "name", config.semantic_backend)),
    ]
    return [f"{role}={name}" for role, name in used if name in PLACEHOLDER_BACKENDS]


@dataclass
class DeliveryConfig:
    # standard11 rather than the condition pipeline's ade20k default: the
    # delivery semantic and proxy encodings are defined over DATA_F.md's 11
    # classes, so predicting them directly avoids a lossy projection.
    depth_backend: str = "mapanything"
    semantic_backend: str = "standard11"
    semantic_refiner: str = "none"
    size: tuple[int, int] = (DELIVERY_WIDTH, DELIVERY_HEIGHT)
    # Frames per model call. This bounds activation memory on the GPU, not the
    # host stacks, which are held whole; see `extract_scene` for their cost.
    chunk_frames: int = 64
    temporal_radius: int = 2
    flow_compensate: bool = True
    # Optical flow is solved at 1/N of the delivery size and scaled back up.
    # Farneback is quadratic in pixel count and a five-frame window needs four
    # flows per frame, so at 720p it otherwise dominates the whole CPU budget.
    flow_downscale: int = 2
    split_hero: bool = True
    color_crf: int = proxy.DEFAULT_COLOR_CRF
    # Forward: near is code 0, far 254, sky 255. Opposite to depth.mp4 on
    # purpose; see the note on `proxy.PROXY_SKY_CODE`.
    inverted_duv_depth: bool = False
    fps: float | None = None
    depth_backend_options: dict = field(default_factory=dict)
    semantic_backend_options: dict = field(default_factory=dict)
    refiner_options: dict = field(default_factory=dict)

    @property
    def taxonomy(self) -> str:
        return _TAXONOMY_OF_BACKEND.get(self.semantic_backend, "cwm12")


def extract_scene(
    video: Path,
    out_dir: Path,
    *,
    annotations: Path | None = None,
    config: DeliveryConfig | None = None,
    depth_backend=None,
    semantic_backend=None,
    refiner=None,
) -> dict:
    """Write one scene directory from one RGB episode.

    Host memory scales with episode length rather than with a window: a
    1800-frame episode holds 6.6 GB of float32 depth, 1.7 GB of labels and 1.7
    GB of flow guide, and the stabilisers allocate their output before freeing
    their input, so peak resident size lands around 20 GB. That is the price of
    running the temporal stages and the protagonist tracker over the whole
    episode at once, which is what makes them equivalent to the batch
    behaviour the tests cover. Size worker count against it.

    Backends may be passed in already constructed so a batch run loads each
    model once rather than once per episode.
    """
    config = config or DeliveryConfig()
    video, out_dir = Path(video), Path(out_dir)
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(video)
    fps = config.fps or (info.fps if info.fps and info.fps > 0 else 24.0)
    width, height = config.size

    depth_backend = depth_backend or get_depth_backend(
        config.depth_backend, **config.depth_backend_options
    )
    semantic_backend = semantic_backend or get_semantic_backend(
        config.semantic_backend, **config.semantic_backend_options
    )
    if refiner is None:
        refiner = get_refiner(config.semantic_refiner, **config.refiner_options)

    inferred = _infer(
        video,
        out_dir / VIDEO_NAMES["color"],
        config,
        fps=fps,
        depth_backend=depth_backend,
        semantic_backend=semantic_backend,
        refiner=refiner,
    )
    if not inferred.metric:
        raise DeliveryError(
            f"{config.depth_backend} returned up-to-scale depth, but the delivery videos "
            "encode absolute metres. ABot's COLMAP model is itself only defined up to a "
            "similarity, so it cannot supply the missing scale either. Use a depth backend "
            "that predicts metric depth."
        )

    depth, labels, guide = inferred.depth, inferred.labels, inferred.guide
    flicker_before = flicker_rate(labels)

    depth = stabilize_depth(
        depth,
        guide_frames=guide,
        radius=config.temporal_radius,
        flow_downscale=config.flow_downscale,
    )
    labels = stabilize_labels(
        labels,
        guide_frames=guide,
        radius=config.temporal_radius,
        flow_downscale=config.flow_downscale,
    )
    del guide
    inferred.guide = None

    labels, hero_info = split_people(
        labels, taxonomy=config.taxonomy, enabled=config.split_hero
    )
    depth, range_stats = apply_range_guard(
        depth, near=DELIVERY_NEAR_METRES, far=DELIVERY_FAR_METRES
    )

    written = _encode(
        out_dir,
        depth,
        labels,
        fps=fps,
        taxonomy=config.taxonomy,
        driving=bool(hero_info.get("driving", False)),
        inverted_duv_depth=config.inverted_duv_depth,
    )
    written["color"] = str(out_dir / VIDEO_NAMES["color"])

    annotation = _copy_annotation(video, out_dir, annotations)

    placeholders = placeholder_backends(depth_backend, semantic_backend, config)
    if placeholders:
        warnings.warn(
            f"{out_dir.name} was produced with placeholder backend(s) "
            f"{', '.join(placeholders)}: the videos are structurally valid but their "
            "content is fabricated and must not be delivered or trained on",
            PlaceholderOutput,
            stacklevel=2,
        )

    names = NAMES_OF_TAXONOMY[config.taxonomy]
    report = {
        "scene": out_dir.name,
        "source_video": str(video),
        "source_size": [info.width, info.height],
        "frames": len(labels),
        "size": [width, height],
        "fps": fps,
        "inference_batches": inferred.batches,
        "config": {**asdict(config), "size": [width, height]},
        "depth": {
            "backend": getattr(depth_backend, "name", config.depth_backend),
            "metric_source": "backend_native",
            "encoding": "h264-logz-gray8",
            "near_metres": DELIVERY_NEAR_METRES,
            "far_metres": DELIVERY_FAR_METRES,
            **range_stats,
            "meta": inferred.depth_meta,
        },
        "semantic": {
            "backend": getattr(semantic_backend, "name", config.semantic_backend),
            "refiner": getattr(refiner, "name", None),
            "taxonomy": config.taxonomy,
            "class_fractions": {
                names[cls]: round(float((labels == cls).mean()), 6) for cls in np.unique(labels)
            },
            "hero_split": hero_info,
            "flicker_before": flicker_before,
            "flicker_after": flicker_rate(labels),
            "meta": inferred.semantic_meta,
        },
        "duv_depth_inverted": config.inverted_duv_depth,
        "placeholder_backends": placeholders,
        "deliverable": not placeholders,
        "videos": written,
        "annotation": annotation,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2))
    return report


@dataclass
class _Inferred:
    depth: np.ndarray  # (N, H, W) float32 metres, invalid = 0
    labels: np.ndarray  # (N, H, W) uint8
    guide: list[np.ndarray] | None  # grayscale frames for flow compensation
    metric: bool
    batches: int
    depth_meta: dict
    semantic_meta: dict


def _infer(
    video: Path,
    color_path: Path,
    config: DeliveryConfig,
    *,
    fps: float,
    depth_backend,
    semantic_backend,
    refiner,
) -> _Inferred:
    """Decode once: write the colour video and run both models as we go.

    One decode pass rather than two. Re-reading the file for the colour track
    would be cheap, but it would also be a second resample of the same source,
    and the delivery set's whole value is that its four streams describe the
    same pixels.
    """
    import cv2

    width, height = config.size
    depth_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    guide: list[np.ndarray] = []
    depth_meta: dict = {}
    semantic_meta: dict = {}
    metric = True
    batches = 0

    color = proxy.open_encoder(
        color_path, width, height, fps, kind="color", crf=config.color_crf
    )
    try:
        for batch in iter_frames(video, size=config.size, chunk=config.chunk_frames):
            for frame in batch:
                color.write(frame)

            depth_result = depth_backend.estimate(batch, cameras=None)
            semantic_result = semantic_backend.segment(batch)
            if refiner is not None:
                semantic_result = refiner.refine(batch, semantic_result)

            depth_parts.append(
                np.where(depth_result.valid_mask(), depth_result.depth, 0.0).astype(np.float32)
            )
            label_parts.append(np.asarray(semantic_result.labels, dtype=np.uint8))
            if config.flow_compensate:
                guide.extend(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in batch)

            metric = metric and depth_result.metric
            depth_meta, semantic_meta = depth_result.meta, semantic_result.meta
            batches += 1
    finally:
        color.close()

    if not depth_parts:
        raise DeliveryError(f"decoded zero frames from {video}")

    return _Inferred(
        depth=np.concatenate(depth_parts),
        labels=np.concatenate(label_parts),
        guide=guide if config.flow_compensate else None,
        metric=metric,
        batches=batches,
        depth_meta=depth_meta,
        semantic_meta=semantic_meta,
    )


def _encode(
    out_dir: Path,
    depth: np.ndarray,
    labels: np.ndarray,
    *,
    fps: float,
    taxonomy: str,
    driving: bool,
    inverted_duv_depth: bool,
) -> dict[str, str]:
    """Write the depth, semantic and duv videos frame by frame."""
    frames, height, width = labels.shape
    standard11 = proxy.to_standard11(labels, taxonomy)

    encoders = {
        stream: proxy.open_encoder(
            out_dir / VIDEO_NAMES[stream], width, height, fps, kind=_ENCODER_KIND[stream]
        )
        for stream in ("depth", "semantic", "duv")
    }
    try:
        for ordinal in range(frames):
            metres = depth[ordinal]
            encoders["depth"].write(proxy.encode_depth_frame(metres))
            encoders["semantic"].write(proxy.encode_semantic_frame(standard11[ordinal]))
            encoders["duv"].write(
                proxy.compose_proxy_frame(
                    metres,
                    standard11[ordinal],
                    driving=driving,
                    inverted_depth=inverted_duv_depth,
                )
            )
    finally:
        errors = []
        for encoder in encoders.values():
            try:
                encoder.close()
            except proxy.EncodeError as error:
                errors.append(str(error))
        if errors:
            raise proxy.EncodeError("; ".join(errors))

    return {stream: str(out_dir / VIDEO_NAMES[stream]) for stream in encoders}


def _copy_annotation(video: Path, out_dir: Path, annotations: Path | None) -> str | None:
    """Place the episode's annotation tar in the scene, verbatim.

    Verbatim because its contents are the dataset's own claims - actions,
    caption, COLMAP model - and repacking them would make this pipeline a
    second source of truth for data it did not produce.
    """
    source = Path(annotations) if annotations is not None else video.parent / "annotations.tar"
    if not source.is_file():
        return None
    target = out_dir / ANNOTATION_NAME
    shutil.copyfile(source, target)
    return str(target)


def scene_dir_for(output_root: Path, index: int) -> Path:
    return Path(output_root) / f"{SCENE_PREFIX}{index:06d}"


# ------------------------------------------------------------------- batching


@dataclass(frozen=True)
class SceneAssignment:
    """One episode's place in the delivered set."""

    index: int
    sample_id: str
    video: Path
    annotations: Path | None

    @property
    def scene(self) -> str:
        return f"{SCENE_PREFIX}{self.index:06d}"


def assign_scenes(
    episodes: list[tuple[str, Path, Path | None]],
) -> list[SceneAssignment]:
    """Number episodes `seg_long_000000` upward, ordered by sample id.

    Sorting by sample id rather than by discovery order makes the numbering a
    function of the input set alone. Two consequences matter: sharded workers
    derive identical numbers without coordinating, and a rerun after adding
    episodes does not silently renumber the ones already delivered - it only
    inserts, which a manifest diff will show.
    """
    ordered = sorted(episodes, key=lambda item: item[0])
    duplicates = {sample for sample, _, _ in ordered if sum(1 for s, _, _ in ordered if s == sample) > 1}
    if duplicates:
        raise DeliveryError(
            f"sample ids are the scene numbering's only key and these repeat: {sorted(duplicates)}"
        )
    return [
        SceneAssignment(index=index, sample_id=sample, video=video, annotations=annotations)
        for index, (sample, video, annotations) in enumerate(ordered)
    ]


def episodes_from_videos(paths: list[Path]) -> list[tuple[str, Path, Path | None]]:
    """Derive sample ids from clip paths, for corpora without a reader.

    ABot names every clip `video.mp4` and distinguishes episodes by the
    directory above it, so the stem alone would collide on the whole corpus.
    """
    episodes = []
    for path in paths:
        path = Path(path)
        sample_id = path.parent.name if path.stem == "video" else path.stem
        tar = path.parent / "annotations.tar"
        episodes.append((sample_id, path, tar if tar.is_file() else None))
    return episodes


MANIFEST_NAME = "scenes_manifest.json"


def write_manifest(output_root: Path, assignments: list[SceneAssignment]) -> Path:
    """Record which episode became which scene.

    Renumbering discards the dataset's own identifiers, so without this the
    delivered set cannot be traced back to the corpus it came from, and a
    problem found in one scene cannot be looked up in the source. Written
    atomically because every sharded worker writes the same content.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenes": [
            {
                "scene": item.scene,
                "sample_id": item.sample_id,
                "source_video": str(item.video),
                "source_annotations": str(item.annotations) if item.annotations else None,
            }
            for item in assignments
        ],
        "count": len(assignments),
    }
    path = output_root / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def read_manifest(output_root: Path) -> list[SceneAssignment]:
    """Rebuild the assignment list from a written manifest."""
    path = Path(output_root) / MANIFEST_NAME
    if not path.is_file():
        raise DeliveryError(f"no {MANIFEST_NAME} under {output_root}; nothing to audit against")
    payload = json.loads(path.read_text())
    return [
        SceneAssignment(
            index=int(entry["scene"].removeprefix(SCENE_PREFIX)),
            sample_id=entry["sample_id"],
            video=Path(entry["source_video"]),
            annotations=Path(entry["source_annotations"]) if entry["source_annotations"] else None,
        )
        for entry in payload["scenes"]
    ]


def audit(output_root: Path, assignments: list[SceneAssignment] | None = None) -> dict:
    """Count what is on disk against what the manifest says should be there.

    A run of 2000 episodes across eight workers takes hours, and the thing an
    operator needs is not the tail of a log but a single answer to "is it done,
    and is what it produced whole". So every scene is re-opened and its videos
    counted, rather than trusting that a directory exists: a worker killed
    mid-encode leaves four files of which some are short, which is exactly the
    failure a marker file would hide.
    """
    output_root = Path(output_root)
    assignments = assignments if assignments is not None else read_manifest(output_root)

    complete: list[str] = []
    incomplete: list[str] = []
    missing: list[str] = []
    total_bytes = 0
    total_frames = 0

    for item in assignments:
        scene_dir = scene_dir_for(output_root, item.index)
        if not scene_dir.is_dir():
            missing.append(item.scene)
            continue
        if already_done(scene_dir):
            complete.append(item.scene)
            total_bytes += sum(p.stat().st_size for p in scene_dir.iterdir() if p.is_file())
            try:
                total_frames += int(json.loads((scene_dir / REPORT_NAME).read_text())["frames"])
            except (ValueError, OSError, KeyError):
                pass
        else:
            incomplete.append(item.scene)

    done = len(complete)
    expected = len(assignments)
    return {
        "output_root": str(output_root),
        "expected": expected,
        "complete": done,
        "incomplete": len(incomplete),
        "missing": len(missing),
        "fraction_done": round(done / expected, 4) if expected else 0.0,
        "frames": total_frames,
        "bytes": total_bytes,
        "gib": round(total_bytes / 2**30, 2),
        "mib_per_scene": round(total_bytes / done / 2**20, 1) if done else None,
        # Projected from what has actually landed, so the estimate sharpens as
        # the run proceeds instead of staying a guess made before it started.
        "projected_total_gib": round(total_bytes / done * expected / 2**30, 1) if done else None,
        "incomplete_scenes": incomplete[:20],
        "missing_scenes": missing[:20],
    }


def deliver_dataset(
    assignments: list[SceneAssignment],
    output_root: Path,
    *,
    config: DeliveryConfig | None = None,
    resume: bool = False,
    on_error: str = "raise",
) -> list[dict]:
    """Run `extract_scene` over many episodes, loading each model exactly once."""
    if on_error not in {"raise", "skip"}:
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    config = config or DeliveryConfig()
    depth_backend = get_depth_backend(config.depth_backend, **config.depth_backend_options)
    semantic_backend = get_semantic_backend(
        config.semantic_backend, **config.semantic_backend_options
    )
    refiner = get_refiner(config.semantic_refiner, **config.refiner_options)

    reports = []
    for item in assignments:
        scene_dir = scene_dir_for(output_root, item.index)
        if resume and already_done(scene_dir):
            reports.append({"scene": item.scene, "skipped": "already complete"})
            continue
        try:
            reports.append(
                extract_scene(
                    item.video,
                    scene_dir,
                    annotations=item.annotations,
                    config=config,
                    depth_backend=depth_backend,
                    semantic_backend=semantic_backend,
                    refiner=refiner,
                )
            )
        except Exception as error:
            # In a run of thousands, one unreadable episode should cost one
            # scene, not the worker's whole shard.
            if on_error == "raise":
                raise
            reports.append({"scene": item.scene, "failed": f"{type(error).__name__}: {error}"})
    return reports


def already_done(scene_dir: Path) -> bool:
    """Whether a previous run left a complete scene here.

    Checked against the report's own frame count rather than mere existence, so
    a worker killed mid-encode leaves something that fails this test and gets
    redone instead of a truncated scene silently entering the dataset.

    Every failure mode has to answer False rather than raise. A half-written
    mp4 has no `moov` atom and cannot be opened at all, and this is the function
    `--resume` consults for every scene: raising here would take down a worker
    over one damaged directory, which is the opposite of what resuming is for.
    """
    scene_dir = Path(scene_dir)
    report_path = scene_dir / REPORT_NAME
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text())
        expected = int(report["frames"])

        for name in VIDEO_NAMES.values():
            path = scene_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                return False
            if probe(path).frames != expected:
                return False
    except (ValueError, OSError, KeyError):
        return False
    return True
