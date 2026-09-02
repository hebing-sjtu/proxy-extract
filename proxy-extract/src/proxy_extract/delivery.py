"""Predict a delivery scene from a plain RGB episode.

`pipeline.extract_clip` writes the `condition_root` that code-world-model
consumes: raw float32 depth and 8-bit IDs on the 336x192 grid its windows are
cut to. This module writes the other deliverable, the one DATA_F.md specifies —
four aligned videos at 1280x720, full episode length, plus the source
annotations:

    seg_000000/
        proxy/
            color.mp4       RGB, libx264 / yuv420p
            depth.mp4       inverted log-z, 8-bit grey, lossless
            semantic.mp4    (R, G, B) = (0, 0, id), lossless RGB
            duv.mp4         R = log-z, G/B = semantic colour, lossless RGB
        annotations.tar     the episode's own annotations, verbatim
        extraction_report.json

The four videos sit under `proxy/` because they are all derived: predicted or
composed by this pipeline at a reduced resolution. What came with the corpus
stays beside that directory, not inside it, so a reader can tell at a glance
which half of a scene is the dataset's own claim and which half is ours.

The 336x192 reduction is deliberately *not* applied here. It throws away 93% of
the pixels and it is cheap to redo from these videos later, so doing it now
would only mean the choice of grid could never be revisited. Nothing else is
held back: every decoded frame is delivered, at the source frame rate.

1280x720 is exactly two-thirds of ABot's 1920x1080, so the resize introduces no
aspect distortion, and it is the resolution `semantic.player`'s priors were
fitted at — so the protagonist split runs on native pixels here rather than on
a reduced grid.

The colour video is encoded from the same decoded frames the models saw, rather
than by handing the source file to ffmpeg separately. Two resamplers do not
agree to the pixel, and a delivery set whose RGB is a fraction of a pixel off
from its own depth is worse than useless for anything that learns a
correspondence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from . import frames, proxy, streaming
from .depth import get_backend as get_depth_backend
from .semantic import get_backend as get_semantic_backend
from .semantic import get_refiner
from .semantic.people import split_people
from .taxonomy import NAMES_OF_TAXONOMY
from .temporal import DEFAULT_MIN_RUN, DEFAULT_RADIUS, suppress_short_runs
from .video import iter_frames, prefetch, probe

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

# Everything this pipeline derives goes here; everything the corpus supplied
# stays in the scene root next to it.
PROXY_DIRNAME = "proxy"

ANNOTATION_NAME = "annotations.tar"
REPORT_NAME = "extraction_report.json"

# Long-form segments: one directory per episode, full length, no window cutting.
SCENE_PREFIX = "seg_"

_TAXONOMY_OF_BACKEND = {"coarse6": "coarse6", "standard11": "standard11"}

# Backends that fabricate their output. They exist so the pipeline can be
# exercised without a GPU, and their videos are structurally indistinguishable
# from real ones — same size, same codecs, same class ids — which is exactly why
# a run using them has to say so in the report rather than leaving a reader to
# infer it from a backend name.
PLACEHOLDER_BACKENDS = frozenset({"synthetic"})


# Failures that say the machine is wrong rather than the episode. The model
# backends import torch and their own packages lazily, inside the first call, so
# a missing dependency does not surface until an episode is already in flight —
# right where `--keep-going` would otherwise swallow it once per episode for the
# length of the corpus.
ENVIRONMENT_ERRORS = (ImportError,)


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
    # Frames decoded and handed to the models at a time. Bounds activation
    # memory on the GPU and nothing else: the host side streams, so this is a
    # throughput knob rather than a memory ceiling, and it is deliberately not
    # part of the resume fingerprint.
    chunk_frames: int = 64
    # Decoded batches kept ready on a background thread, so the models are not
    # waiting on the source filesystem between forwards.
    prefetch_batches: int = 2
    temporal_radius: int = DEFAULT_RADIUS
    temporal_min_run: int = DEFAULT_MIN_RUN
    # Frames released per flow pass. Larger repeats slightly less optical flow
    # at the window seams and holds proportionally more frames while doing it.
    stabilise_block: int = streaming.DEFAULT_BLOCK
    # Frame writes are handed to this many threads so the models are not
    # waiting behind PNG encoding and a filesystem. 0 writes inline.
    writer_threads: int = 4
    # Which per-frame directories survive the encode, if any. Per stream rather
    # than all-or-nothing because they are not worth the same:
    #
    #   depth     float16 metres. The only one the videos cannot reproduce -
    #             depth.mp4 quantises this onto 8 bits, losing about 60x.
    #   color     lossless. color.mp4 is CRF 16, so this is slightly better.
    #   semantic  the same ids semantic.mp4 already carries losslessly.
    #   duv       derivable outright from depth and semantic.
    #
    # At 1280x720 the two array streams cost 2.64 MiB a frame between them, or
    # 4.6 GiB per 1800-frame episode, and that is before the images. Over a
    # corpus this is measured in terabytes, so it is worth saying which ones
    # are actually wanted.
    keep_frames: tuple[str, ...] = frames.STREAMS
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
    # Where to say what the worker is doing, if anywhere. A scene takes minutes
    # and a shard takes days, so a worker that says nothing is indistinguishable
    # from a wedged one — which is a diagnosis that has cost hours. The CLI
    # supplies a timestamped printer; library callers and tests get silence.
    progress: "Callable[[str], None] | None" = None
    # Seconds between within-scene heartbeats. Only the beats are throttled;
    # stage transitions always print.
    progress_interval: float = 30.0
    # Seconds between checkpoints of the inference stage. Each one waits for
    # every queued write to land before recording the frame count, so this is
    # also the interval on which the models stop and wait for the filesystem -
    # and, on a restart, the most work that can be lost. A minute costs under a
    # percent of a stage that runs for tens of them.
    checkpoint_interval: float = 60.0

    @property
    def taxonomy(self) -> str:
        return _TAXONOMY_OF_BACKEND.get(self.semantic_backend, "cwm12")


# Fields that describe how the run is driven rather than what it produces, and
# that JSON cannot hold anyway.
_UNREPORTABLE = ("progress", "progress_interval")


def _reportable(config: DeliveryConfig) -> dict:
    """The config as the report records it: settings only, no machinery."""
    return {key: value for key, value in asdict(config).items() if key not in _UNREPORTABLE}


class _Progress:
    """Says what a worker is doing, rarely enough to leave the log readable.

    Stage transitions always print, since there are four of them per episode.
    Within a stage the caller beats on every frame and this drops all but one
    every `interval` seconds - a 1800-frame episode would otherwise be 1800
    lines, times 2000 episodes, times 64 shards.
    """

    def __init__(self, config: DeliveryConfig, scene: str) -> None:
        self.sink = config.progress
        self.interval = max(float(config.progress_interval), 0.0)
        self.scene = scene
        self._last = 0.0

    def say(self, message: str) -> None:
        if self.sink is not None:
            self.sink(f"{self.scene} {message}")

    def beat(self, message: str) -> None:
        """Print at most one of these per interval, and always the first."""
        if self.sink is None:
            return
        now = time.monotonic()
        if now - self._last < self.interval:
            return
        self._last = now
        self.sink(f"{self.scene} {message}")


class _Phases:
    """Where a stage's wall clock went, as shares of it.

    A worker holding VRAM at 0% utilisation is waiting on the source, waiting
    on the output filesystem, or actually computing, and the three call for
    completely different fixes - more workers, fewer workers, faster storage.
    From outside the process they are indistinguishable, and diagnosing one as
    another has already cost this project days. So the stage measures itself
    and puts the answer in its own progress line.

    Wall time, not CPU time: the question is what the pipeline is waiting for,
    and a thread blocked on a network filesystem burns no CPU at all.
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.totals: dict[str, float] = {}

    @contextmanager
    def timing(self, phase: str):
        mark = time.monotonic()
        try:
            yield
        finally:
            self.totals[phase] = self.totals.get(phase, 0.0) + (time.monotonic() - mark)

    @property
    def elapsed(self) -> float:
        return max(time.monotonic() - self.started, 1e-9)

    def shares(self, **extra: float) -> str:
        """The phases as percentages, with whatever is left over named.

        `other` is not padding: in this stage it is the temporal stabilisation,
        which is CPU work nobody thinks about until it is the majority.
        """
        named = {**self.totals, **extra}
        rest = self.elapsed - sum(named.values())
        parts = [f"{name} {value / self.elapsed:.0%}" for name, value in named.items()]
        parts.append(f"other {max(rest, 0.0) / self.elapsed:.0%}")
        return ", ".join(parts)


def _pace(frames_done: int, total: int | None, clock: _Phases, blocked: float) -> str:
    """One phrase: how fast, how much longer, and what the time went on."""
    rate = frames_done / clock.elapsed
    eta = ""
    if total and rate > 0 and total > frames_done:
        seconds = (total - frames_done) / rate
        eta = f", eta {seconds / 60:.0f}m" if seconds >= 60 else f", eta {seconds:.0f}s"
    return f"{rate:.1f} f/s{eta} ({clock.shares(write=blocked)})"


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
    """Write one scene directory from one RGB episode, in resumable stages.

    An episode passes through three of them, and each one's output is on disk
    before the next starts:

        infer   decode, predict, stabilise; write color/ and depth/ frames
        derive  suppress runs, split the protagonist; write semantic/ and duv/
        encode  read the four frame directories back into four videos

    So a worker killed at frame 1700 of 1800 resumes at frame 1700. That used
    to cost the whole episode, which over a corpus this size is the difference
    between a run that tolerates a node going away and one that does not.

    Only `derive` needs an episode-sized array, and only of labels: run
    suppression and the protagonist tracker both have to see every frame before
    they can decide anything. Everything else streams, so peak resident size is
    a few GB rather than the ~40 that used to cap workers per GPU at three.

    Backends may be passed in already constructed so a batch run loads each
    model once rather than once per episode.
    """
    config = config or DeliveryConfig()
    video, out_dir = Path(video), Path(out_dir)
    started = time.time()

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

    out_dir.mkdir(parents=True, exist_ok=True)
    proxy_dir_for(out_dir).mkdir(parents=True, exist_ok=True)
    frames.make_dirs(out_dir)
    state = _open_state(out_dir, _fingerprint(config, video, fps))

    progress = _Progress(config, out_dir.name)
    progress.say(f"infer: {video}")
    state = _stage_infer(
        video,
        out_dir,
        config,
        state=state,
        depth_backend=depth_backend,
        semantic_backend=semantic_backend,
        refiner=refiner,
        progress=progress,
        source_frames=info.frames or None,
    )
    if not state["metric"]:
        raise DeliveryError(
            f"{config.depth_backend} returned up-to-scale depth, but the delivery videos "
            "encode absolute metres. ABot's COLMAP model is itself only defined up to a "
            "similarity, so it cannot supply the missing scale either. Use a depth backend "
            "that predicts metric depth."
        )

    progress.say(f"derive: {state['frames']} frames")
    state = _stage_derive(out_dir, config, state=state, progress=progress)
    progress.say("encode: four videos")
    written = _stage_encode(out_dir, config, fps=fps, state=state)
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

    report = {
        "scene": out_dir.name,
        "source_video": str(video),
        "source_size": [info.width, info.height],
        "frames": state["frames"],
        "size": [width, height],
        "fps": fps,
        "inference_batches": state["batches"],
        "config": {**_reportable(config), "size": [width, height]},
        "depth": {
            "backend": getattr(depth_backend, "name", config.depth_backend),
            "metric_source": "backend_native",
            "encoding": "h264-logz-gray8",
            "near_metres": DELIVERY_NEAR_METRES,
            "far_metres": DELIVERY_FAR_METRES,
            **state["range"],
            "meta": state["depth_meta"],
        },
        "semantic": {
            "backend": getattr(semantic_backend, "name", config.semantic_backend),
            "refiner": getattr(refiner, "name", None),
            "taxonomy": config.taxonomy,
            "class_fractions": state["class_fractions"],
            "hero_split": state["hero_split"],
            "flicker_before": state["flicker_before"],
            "flicker_after": state["flicker_after"],
            "meta": state["semantic_meta"],
        },
        "duv_depth_inverted": config.inverted_duv_depth,
        "placeholder_backends": placeholders,
        "deliverable": not placeholders,
        "videos": written,
        "frames_kept": sorted(config.keep_frames),
        "annotation": annotation,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2))

    # Last, so that everything above can be resumed if it fails. The report is
    # the marker that says this scene is finished; nothing may outlive it.
    frames.clear_stage(out_dir)
    frames.keep_only(out_dir, config.keep_frames)
    return report


# --------------------------------------------------------------------- stages

STATE_NAME = "state.json"

# The one piece of checkpoint state that is an array rather than a number: the
# last labels the flicker meter compared, which the next process needs as the
# left-hand side of its first comparison.
FLICKER_PREVIOUS = "flicker_previous.npy"


def _state_path(scene_dir: Path) -> Path:
    return frames.stage_dir_for(scene_dir) / STATE_NAME


def _fingerprint(config: DeliveryConfig, video: Path, fps: float) -> str:
    """What has to match for frames already on disk to be reusable.

    Deliberately not everything in the config. `chunk_frames`,
    `stabilise_block`, `writer_threads` and `keep_frames` change how the work is
    divided and never what it produces, so an operator who restarts a run with
    a bigger batch to fill a larger GPU keeps the frames the smaller one wrote.
    Anything that would change a pixel is in here.
    """
    payload = {
        "video": str(video),
        "fps": fps,
        "size": list(config.size),
        "depth_backend": config.depth_backend,
        "depth_backend_options": config.depth_backend_options,
        "semantic_backend": config.semantic_backend,
        "semantic_backend_options": config.semantic_backend_options,
        "semantic_refiner": config.semantic_refiner,
        "refiner_options": config.refiner_options,
        "temporal_radius": config.temporal_radius,
        "temporal_min_run": config.temporal_min_run,
        "flow_compensate": config.flow_compensate,
        "flow_downscale": config.flow_downscale,
        "split_hero": config.split_hero,
        "inverted_duv_depth": config.inverted_duv_depth,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _open_state(scene_dir: Path, fingerprint: str) -> dict:
    """Read the working state, discarding it if it describes a different run.

    Resuming onto frames produced by different settings would deliver an
    episode that is half one thing and half another, and nothing downstream
    could detect it - the videos would be the right length and the right
    format. So a fingerprint mismatch throws the frames away rather than
    trying to reconcile them.
    """
    path = _state_path(scene_dir)
    if path.is_file():
        try:
            state = json.loads(path.read_text())
            if state.get("fingerprint") == fingerprint:
                return state
        except (ValueError, OSError):
            pass
    for stream in frames.ALL_STREAMS:
        frames.drop_stream(scene_dir, stream)
    frames.make_dirs(scene_dir)
    return {
        "fingerprint": fingerprint,
        "stage": "infer",
        "frames": 0,
        "batches": 0,
        "metric": True,
        "depth_meta": {},
        "semantic_meta": {},
    }


def _save_state(scene_dir: Path, state: dict) -> None:
    path = _state_path(scene_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _stage_infer(
    video: Path,
    scene_dir: Path,
    config: DeliveryConfig,
    *,
    state: dict,
    depth_backend,
    semantic_backend,
    refiner,
    progress: "_Progress | None" = None,
    source_frames: int | None = None,
) -> dict:
    """Decode once, predict, stabilise, and write colour and depth frames.

    One decode pass rather than two. Re-reading the file for the colour track
    would be cheap, but it would also be a second resample of the same source,
    and the delivery set's whole value is that its four streams describe the
    same pixels.

    Restarting re-runs the models over `temporal_radius` frames before the
    resume point. Those frames are already written and their output is thrown
    away; what they are for is giving the first frame that is *not* written the
    same left-hand neighbours it would have had in an uninterrupted run, so the
    seam is invisible in the result rather than merely small.
    """
    import cv2

    if state["stage"] != "infer":
        return state

    progress = progress or _Progress(config, scene_dir.name)
    written_streams = ("color", "depth", frames.STAGING_STREAM)
    # Frames on disk are only usable as far as the last checkpoint: the range
    # guard's and the flicker meter's totals were saved there, and frames
    # written after it are ones those totals have never seen. Counting them as
    # done would drop them from both statistics silently. So the checkpoint is
    # the resume point, and anything past it is written again.
    done = min(
        frames.complete_through(scene_dir, written_streams),
        int(state.get("frames", 0)) if "range_state" in state else 0,
    )
    if done:
        progress.say(f"resuming from frame {done}")
    for stream in written_streams:
        frames.discard_from(scene_dir, stream, done)

    guard = streaming.RangeGuard(near=DELIVERY_NEAR_METRES, far=DELIVERY_FAR_METRES)
    flicker = streaming.FlickerMeter()
    if done and "range_state" in state:
        guard.restore(state["range_state"])
        flicker.restore(state["flicker_state"])
        flicker.seed(frames.read_stage_array(scene_dir, FLICKER_PREVIOUS))

    window = streaming.WindowStabiliser(
        radius=config.temporal_radius,
        flow_downscale=config.flow_downscale,
        block=config.stabilise_block,
        flow_compensate=config.flow_compensate,
    )
    infer_from = max(done - config.temporal_radius, 0)

    metric = bool(state["metric"])
    batches = int(state["batches"])
    depth_meta: dict = dict(state["depth_meta"])
    semantic_meta: dict = dict(state["semantic_meta"])
    decoded = 0
    total = done
    clock = _Phases()
    checkpointed = time.monotonic()

    with _FrameWriter(scene_dir, config.writer_threads) as writer:

        def keep(
            ordinal: int, metres: np.ndarray, labels: np.ndarray, raw_labels: np.ndarray
        ) -> None:
            nonlocal total
            if ordinal < done:
                return
            # Measured here rather than where the models produced it, so that
            # the meter and the range guard advance in step with the frames on
            # disk. Inference runs a window ahead of what has been released, so
            # counting at that point would checkpoint statistics covering
            # frames the next process is going to compute again.
            flicker.push(raw_labels)
            writer.array("depth", ordinal, guard.apply(metres))
            writer.array(frames.STAGING_STREAM, ordinal, labels)
            total = max(total, ordinal + 1)

        decoding = iter(
            prefetch(
                iter_frames(video, size=config.size, chunk=config.chunk_frames),
                depth=config.prefetch_batches,
            )
        )
        while True:
            # Timed separately from the models because a batch that is not
            # ready means the source filesystem is the constraint, and no
            # amount of GPU answers that.
            with clock.timing("read"):
                batch = next(decoding, None)
            if batch is None:
                break

            start, decoded = decoded, decoded + len(batch)
            if decoded <= infer_from:
                continue

            # A batch straddling the resume point is decoded whole and inferred
            # from the point onward; `iter_frames` will not seek, deliberately.
            offset = max(infer_from - start, 0)
            batch = batch[offset:]
            first = start + offset

            for index, frame in enumerate(batch):
                if first + index >= done:
                    writer.image("color", first + index, frame)

            with clock.timing("model"):
                depth_result = depth_backend.estimate(batch, cameras=None)
                semantic_result = semantic_backend.segment(batch)
                if refiner is not None:
                    semantic_result = refiner.refine(batch, semantic_result)

            raw_depth = np.where(
                depth_result.valid_mask(), depth_result.depth, 0.0
            ).astype(np.float32)
            raw_labels = np.asarray(semantic_result.labels, dtype=np.uint8)
            metric = metric and depth_result.metric
            depth_meta, semantic_meta = depth_result.meta, semantic_result.meta
            batches += 1

            for index in range(len(batch)):
                guides = (
                    cv2.cvtColor(batch[index], cv2.COLOR_RGB2GRAY)
                    if config.flow_compensate
                    else None
                )
                # The window numbers what it was given, and it was given frames
                # from `infer_from` on, not from the episode's start.
                for ordinal, metres, labels, raw in window.push(
                    raw_depth[index], raw_labels[index], guides
                ):
                    keep(infer_from + ordinal, metres, labels, raw)

            # Frames read, not frames written. The stabiliser holds a block
            # before it releases anything, so the written count sits at zero
            # for the first `stabilise_block` frames of every episode - a
            # progress line that reads as a hang for exactly as long as the
            # operator is most likely to be watching it.
            progress.beat(
                f"infer {decoded}/{source_frames or '?'} frames, "
                f"{_pace(decoded - infer_from, source_frames, clock, writer.blocked)}"
            )
            # Not every batch. A checkpoint has to wait for every queued write
            # to land before it can record a frame count, so doing it per batch
            # stops the models on the filesystem twenty-eight times an episode
            # for no gain: what it buys back is bounded by this interval, and
            # what it costs is however long the slowest write takes.
            if time.monotonic() - checkpointed >= config.checkpoint_interval:
                writer.drain()
                if flicker.previous is not None:
                    frames.write_stage_array(scene_dir, FLICKER_PREVIOUS, flicker.previous)
                _save_state(
                    scene_dir,
                    {
                        **state,
                        "frames": total,
                        "batches": batches,
                        "metric": metric,
                        "depth_meta": depth_meta,
                        "semantic_meta": semantic_meta,
                        "range_state": guard.state(),
                        "flicker_state": flicker.state(),
                        "range": guard.stats(),
                    },
                )
                checkpointed = time.monotonic()

        for ordinal, metres, labels, raw in window.close():
            keep(infer_from + ordinal, metres, labels, raw)

    if total == 0:
        raise DeliveryError(f"decoded zero frames from {video}")

    state = {
        **state,
        "stage": "derive",
        "frames": total,
        "batches": batches,
        "metric": metric,
        "depth_meta": depth_meta,
        "semantic_meta": semantic_meta,
        "range": guard.stats(),
        "range_state": guard.state(),
        "flicker_state": flicker.state(),
        "flicker_before": flicker.rate,
    }
    _save_state(scene_dir, state)
    return state


def _stage_derive(
    scene_dir: Path,
    config: DeliveryConfig,
    *,
    state: dict,
    progress: "_Progress | None" = None,
) -> dict:
    """Finish the labels, then write the semantic and duv frames.

    The two steps here are the ones that cannot stream. Run suppression asks
    how long a run of a label lasts, and a run has no length until the episode
    ends; the protagonist tracker has to compare every person track in the clip
    before it can say which is the protagonist. Both read labels only, so this
    holds a byte per pixel per frame - 1.7 GB for a 1800-frame episode - rather
    than the depth stack that used to dominate.
    """
    if state["stage"] not in {"derive", "encode"}:
        raise DeliveryError(f"cannot derive from stage {state['stage']!r}")

    progress = progress or _Progress(config, scene_dir.name)
    count = int(state["frames"])
    if state["stage"] == "derive":
        labels = frames.read_stack(scene_dir, frames.STAGING_STREAM, count)
        # In place because this stack was read for these two steps and nothing
        # else; a defensive copy of it is 1.7 GB that no one reads.
        labels = suppress_short_runs(
            labels, min_run=config.temporal_min_run, in_place=True
        )
        labels, hero_info = split_people(
            labels, taxonomy=config.taxonomy, enabled=config.split_hero
        )
        standard11 = proxy.to_standard11(labels, config.taxonomy)
        driving = bool(hero_info.get("driving", False))

        state = {
            **state,
            "hero_split": hero_info,
            "flicker_after": streaming.flicker_rate_of(labels),
            "class_fractions": streaming.class_fractions(
                labels, NAMES_OF_TAXONOMY[config.taxonomy]
            ),
        }
        del labels

        done = frames.complete_through(scene_dir, ("semantic", "duv"))
        for stream in ("semantic", "duv"):
            frames.discard_from(scene_dir, stream, done)

        clock = _Phases()
        with _FrameWriter(scene_dir, config.writer_threads) as writer:
            for ordinal in range(done, count):
                progress.beat(
                    f"derive {ordinal}/{count} frames, "
                    f"{_pace(ordinal - done, count, clock, writer.blocked)}"
                )
                with clock.timing("read"):
                    metres = frames.read_array(scene_dir, "depth", ordinal)
                writer.array("semantic", ordinal, standard11[ordinal])
                writer.image(
                    "duv",
                    ordinal,
                    proxy.compose_proxy_frame(
                        metres,
                        standard11[ordinal],
                        driving=driving,
                        inverted_depth=config.inverted_duv_depth,
                    ),
                )

        state = {**state, "stage": "encode"}
        _save_state(scene_dir, state)
        frames.drop_stream(scene_dir, frames.STAGING_STREAM)
    return state


def _stage_encode(
    scene_dir: Path, config: DeliveryConfig, *, fps: float, state: dict
) -> dict[str, str]:
    """Read the four frame directories back into the four delivery videos.

    Not resumable, unlike the two stages before it: an mp4 cannot be appended
    to, and there is nothing to gain by pretending otherwise. It is also the
    cheapest stage by a wide margin, because every pixel it writes has already
    been decided.
    """
    proxy_dir = proxy_dir_for(scene_dir)
    count = int(state["frames"])
    width, height = config.size

    encoders = {
        stream: proxy.open_encoder(
            proxy_dir / VIDEO_NAMES[stream],
            width,
            height,
            fps,
            kind=_ENCODER_KIND[stream],
            crf=config.color_crf if stream == "color" else None,
        )
        for stream in VIDEO_NAMES
    }
    try:
        for ordinal in range(count):
            encoders["color"].write(frames.read_image(scene_dir, "color", ordinal))
            encoders["depth"].write(
                proxy.encode_depth_frame(frames.read_array(scene_dir, "depth", ordinal))
            )
            encoders["semantic"].write(
                proxy.encode_semantic_frame(frames.read_array(scene_dir, "semantic", ordinal))
            )
            encoders["duv"].write(frames.read_image(scene_dir, "duv", ordinal))
    finally:
        errors = []
        for encoder in encoders.values():
            try:
                encoder.close()
            except proxy.EncodeError as error:
                errors.append(str(error))
        if errors:
            raise proxy.EncodeError("; ".join(errors))

    return {stream: str(proxy_dir / VIDEO_NAMES[stream]) for stream in VIDEO_NAMES}


class _FrameWriter:
    """Hand frame writes to threads so the models are not waiting on a disk.

    Writing a 1800-frame episode is some 3600 PNG encodes and 3600 array
    writes, and in the inference stage every one of them falls between two
    model calls. PNG encoding and file writes both release the GIL, so moving
    them off the calling thread hides nearly all of it behind the forward pass
    that follows.

    The queue is bounded, and that is not a detail. A queued write holds its
    frame alive, so an unbounded queue turns into a copy of the episode in host
    memory - which is the exact cost the streaming rewrite exists to avoid, and
    it would arrive silently, as memory growth rather than as a wrong answer.
    """

    # Enough outstanding writes to keep every thread busy across a slow call,
    # and few enough that what they pin is measured in frames.
    QUEUE_PER_THREAD = 8

    def __init__(self, scene_dir: Path, threads: int) -> None:
        from collections import deque

        self.scene_dir = Path(scene_dir)
        self.threads = max(int(threads), 0)
        self.limit = max(self.threads * self.QUEUE_PER_THREAD, 1)
        self._pool = None
        self._pending: deque = deque()
        # Seconds the caller spent waiting on this writer, which is the only
        # honest measure of what the output filesystem is costing: once the
        # queue is full, back-pressure arrives as a slow `image()` call in the
        # middle of the stabiliser rather than as anything named "writing".
        self.blocked = 0.0

    def __enter__(self) -> "_FrameWriter":
        if self.threads:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=self.threads)
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self.drain()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

    def _submit(self, fn, *args) -> None:
        if self._pool is None:
            mark = time.monotonic()
            fn(*args)
            self.blocked += time.monotonic() - mark
            return
        if len(self._pending) >= self.limit:
            mark = time.monotonic()
            while len(self._pending) >= self.limit:
                self._pending.popleft().result()
            self.blocked += time.monotonic() - mark
        self._pending.append(self._pool.submit(fn, *args))

    def array(self, stream: str, ordinal: int, array: np.ndarray) -> None:
        self._submit(frames.write_array, self.scene_dir, stream, ordinal, array)

    def image(self, stream: str, ordinal: int, rgb: np.ndarray) -> None:
        self._submit(frames.write_image, self.scene_dir, stream, ordinal, rgb)

    def drain(self) -> None:
        """Block until every queued write has landed, re-raising the first failure.

        What makes a checkpoint honest: when the state file records N frames, N
        frames are on disk rather than queued behind a slow filesystem.
        """
        if not self._pending:
            return
        mark = time.monotonic()
        while self._pending:
            self._pending.popleft().result()
        self.blocked += time.monotonic() - mark


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


def proxy_dir_for(scene_dir: Path) -> Path:
    """Where a scene's four derived videos live."""
    return Path(scene_dir) / PROXY_DIRNAME


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
    """Number episodes `seg_000000` upward, ordered by sample id.

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
    problem found in one scene cannot be looked up in the source.

    Every sharded worker writes this, with identical content, within a second
    of the others starting - so the write has to be atomic *and* the scratch
    file has to be unique per call. A shared scratch name is not merely untidy:
    two workers open it, the first renames it into place, and the second's
    rename fails on a file that no longer exists, killing that shard before it
    has read a single frame. On a 64-way launch that quietly costs a handful of
    shards every time, and the survivors look like an uneven GPU assignment.

    `mkstemp` rather than the pid, because the uniqueness wanted here is per
    call: nothing about this function requires its callers to be in different
    processes, and a name that is only unique per process would make that an
    unwritten rule with a nasty failure mode.
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
    handle, scratch = tempfile.mkstemp(dir=output_root, prefix=f"{MANIFEST_NAME}.", suffix=".tmp")
    tmp = Path(scratch)
    try:
        with os.fdopen(handle, "w") as file:
            json.dump(payload, file, indent=2)
        os.replace(tmp, path)
    finally:
        # Gone already on the happy path; this is for a write that raised.
        tmp.unlink(missing_ok=True)
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
            # rglob, not iterdir: the videos are a directory down from here.
            total_bytes += sum(p.stat().st_size for p in scene_dir.rglob("*") if p.is_file())
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


SCENE_STATES = ("complete", "incomplete", "missing")


def list_scenes(
    output_root: Path,
    which: str = "complete",
    assignments: list[SceneAssignment] | None = None,
) -> list[str]:
    """Scene names in one state, in delivery order, for feeding to another tool.

    Separate from `audit` rather than a field of it, because the two are asked
    very different questions. `audit` answers "how far along is this", and pays
    for it by stat-ing every file of every finished scene - which at 1800 frames
    across four per-frame streams is some seven thousand files a scene, and at
    two thousand scenes is a long walk. This answers "which ones can I take
    now", wants none of those sizes, and is asked while the run is still going.

    Plain names on purpose: `seg_000000` is both the directory relative to the
    output root and what `rsync --files-from` wants to read.
    """
    if which not in SCENE_STATES:
        raise DeliveryError(f"unknown scene state {which!r}; expected one of {SCENE_STATES}")

    output_root = Path(output_root)
    assignments = assignments if assignments is not None else read_manifest(output_root)

    names = []
    for item in assignments:
        scene_dir = scene_dir_for(output_root, item.index)
        if not scene_dir.is_dir():
            state = "missing"
        else:
            state = "complete" if already_done(scene_dir) else "incomplete"
        if state == which:
            names.append(item.scene)
    return names


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

    say = config.progress or (lambda _message: None)
    say(f"{len(assignments)} episodes to do")

    reports = []
    for position, item in enumerate(assignments, start=1):
        scene_dir = scene_dir_for(output_root, item.index)
        if resume and already_done(scene_dir):
            say(f"{item.scene} [{position}/{len(assignments)}] already complete")
            reports.append({"scene": item.scene, "skipped": "already complete"})
            continue
        started = time.time()
        say(f"{item.scene} [{position}/{len(assignments)}] start")
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
            say(
                f"{item.scene} done in {time.time() - started:.0f}s, "
                f"{reports[-1]['frames']} frames"
            )
        except ENVIRONMENT_ERRORS as error:
            # Not survivable and not per-episode: the backend's dependency is
            # missing, so every remaining episode fails the same way. Skipping
            # would burn the whole corpus and report a run that produced
            # nothing, which is how a missing `mapanything` once cost 1800
            # episodes' worth of wall time.
            raise DeliveryError(
                f"{type(error).__name__}: {error}\n"
                f"This is an environment problem, not a problem with {item.scene}, so the "
                "run stops here rather than failing identically on every remaining episode. "
                "--keep-going deliberately does not cover it."
            ) from error
        except Exception as error:
            # In a run of thousands, one unreadable episode should cost one
            # scene, not the worker's whole shard.
            if on_error == "raise":
                raise
            say(f"{item.scene} FAILED: {type(error).__name__}: {error}")
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

        proxy_dir = proxy_dir_for(scene_dir)
        for name in VIDEO_NAMES.values():
            path = proxy_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                return False
            if probe(path).frames != expected:
                return False
    except (ValueError, OSError, KeyError):
        return False
    return True
