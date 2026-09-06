"""Short training clips cut out of the delivered long segments.

The delivery set is whole episodes at their own frame rate, because that is the
form that keeps every option open. This module spends those options: it cuts
each episode into a few short, disjoint clips in the shape an H3 SFT sample
takes, and once cut, the choices below cannot be revisited without cutting
again.

    clip_000000/
        target/rgb.mp4        1344x768, 124 frames at 24 fps
        target/anchor.png     1344x768, the target's own frame 0
        proxy/duv.mp4         336x192, the same 124 frames, lossless
        annotations/          this window's slice of the episode's own claims
        clip_report.json      which episode, which frames, and every setting

Three things here are worth reading before trusting the output.

**The frame rate is changed by dropping frames, not by relabelling them.** The
episodes are 30 fps and the clips are 24, so output frame k is source frame
round(k * 30/24): four kept out of every five. Writing 124 consecutive source
frames and calling the result 24 fps would have been one line shorter and would
have made every clip a 1.25x slow motion of what actually happened - a defect
that is invisible in any single frame and fatal to anything learning dynamics.

**The DUV is recomposed at 336x192, not resized to it.** Its red channel is a
log-depth code and its green and blue are a class palette, so interpolating it
averages codes that mean nothing in between and paints classes the segmenter
never predicted. Depth is reduced by median and labels by majority vote, and the
DUV is composed from the results, so every pixel of it is a value that occurred.

**The DUV grid is the target grid over 4x4 blocks.** Both start from the same
delivered frames and reach 1344x768 by nearest-neighbour, which invents no
values, and the DUV is reduced from there by exactly four. So DUV pixel (x, y)
covers target pixels (4x..4x+3, 4y..4y+3) - not approximately, exactly. That is
also the answer to why the target is the delivered 1280x720 colour upscaled 5%
rather than the sharper 1920x1080 source re-decoded: a target resampled through
a different chain than its own control signal loses this property, and
DATA_F.md is explicit that the alignment is worth more than the sharpness.
`--target-from source` takes the other trade for anyone who disagrees.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contract, frames, proxy
from .video import probe

CLIP_PREFIX = "clip_"
CLIPS_PER_SCENE = 5

# One code-world-model window, so a clip is exactly one sample rather than
# something that has to be windowed again downstream. See contract.py.
CLIP_FRAMES = contract.WINDOW_FRAMES
CLIP_FPS = 24.0

# Both must be multiples of 32 for the encoder's latent grid, and both are 1.75
# rather than the source's 16:9 - a 1.6% horizontal squeeze that is applied to
# the target and the DUV alike, so they still describe the same pixels.
TARGET_WIDTH = 1344
TARGET_HEIGHT = 768
DUV_WIDTH = contract.CONDITION_WIDTH
DUV_HEIGHT = contract.CONDITION_HEIGHT

TARGET_DIRNAME = "target"
PROXY_DIRNAME = "proxy"
ANNOTATION_DIRNAME = "annotations"
TARGET_NAME = "rgb.mp4"
ANCHOR_NAME = "anchor.png"
DUV_NAME = "duv.mp4"
CLIP_REPORT_NAME = "clip_report.json"
CLIPS_MANIFEST_NAME = "clips_manifest.json"


class ClipError(RuntimeError):
    pass


@dataclass(frozen=True)
class Window:
    """One clip's slice of one episode, as source frame ordinals."""

    scene: str
    index: int
    ordinals: tuple[int, ...]

    @property
    def start(self) -> int:
        return self.ordinals[0]

    @property
    def stop(self) -> int:
        """One past the last source frame this window reads."""
        return self.ordinals[-1] + 1


def plan_windows(
    scene: str,
    source_frames: int,
    *,
    source_fps: float,
    count: int = CLIPS_PER_SCENE,
    length: int = CLIP_FRAMES,
    fps: float = CLIP_FPS,
) -> list[Window]:
    """Where in an episode the clips come from, evenly spread and disjoint.

    The episode is cut into `count` equal buckets and one window is centred in
    each. Even rather than random because the alternative buys diversity this
    corpus does not need - an episode is one continuous walk, so windows a
    third of a minute apart are already unalike - and costs the property that
    matters more here: which frames a clip came from is a function of the
    episode alone, so a re-run reproduces the set exactly and a clip can be
    traced back without consulting a seed.

    Centring is what keeps them apart. Packing from the start would leave every
    gap at the end of the episode, and windows that share a boundary are two
    halves of one sample as far as anything learning from them is concerned.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if source_fps <= 0 or fps <= 0:
        raise ValueError(f"frame rates must be positive, got {source_fps} and {fps}")

    step = source_fps / fps
    if step < 1.0 - 1e-9:
        raise ClipError(
            f"{scene} is {source_fps:g} fps and the clips are {fps:g}: making a faster "
            "stream out of a slower one needs frames that were never recorded"
        )

    # The source frames one window spans, which is more than its length
    # whenever frames are being dropped: 124 output frames at 24 fps reach
    # across 155 source frames at 30.
    span = round((length - 1) * step) + 1
    bucket = source_frames // count
    if span > bucket:
        raise ClipError(
            f"{scene} has {source_frames} frames, which is {count} buckets of {bucket}, "
            f"but one {length}-frame clip at {fps:g} fps spans {span} of them. Ask for "
            f"fewer clips per episode, or shorter ones."
        )

    windows = []
    for index in range(count):
        start = index * bucket + (bucket - span) // 2
        ordinals = tuple(start + round(k * step) for k in range(length))
        if ordinals[-1] >= source_frames:  # pragma: no cover - guarded by span
            raise ClipError(f"{scene} window {index} runs past frame {source_frames}")
        windows.append(Window(scene=scene, index=index, ordinals=ordinals))
    return windows


def clip_name(scene: str, index: int) -> str:
    """`seg_000123` window 2 becomes `clip_000123_2`.

    The episode's number is kept rather than renumbered globally, because a
    clip's most common question is which segment it came from and the second
    most common is which of that segment's clips it is. A flat global counter
    answers neither without the manifest.
    """
    return f"{CLIP_PREFIX}{scene.split('_', 1)[-1]}_{index}"


# --------------------------------------------------------------------- pixels


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a frame to the target box, whatever direction that is.

    INTER_AREA is right when shrinking and degenerates to something close to
    nearest-neighbour when growing, which is exactly the wrong way round for a
    5% upscale, so the direction has to be chosen rather than assumed.
    """
    import cv2

    if image.shape[1] == width and image.shape[0] == height:
        return image
    growing = width * height > image.shape[1] * image.shape[0]
    interpolation = cv2.INTER_LANCZOS4 if growing else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _to_duv_grid(plane: np.ndarray, reduce) -> np.ndarray:
    """Reduce one 1280x720 plane onto the 336x192 DUV grid, via the target's.

    The delivered size is not a multiple of the DUV's - 720/192 is 3.75 - and
    `contract`'s reductions quietly fall back to nearest-neighbour when the
    factor is not integral, which samples one pixel in fifteen and drops
    whatever it did not land on. Going through 1344x768 first fixes that: the
    step to it is nearest-neighbour, so no value is invented, and from there the
    factor is exactly 4, so the median and the vote run as intended.

    It also buys the alignment: after this, one DUV pixel is exactly one 4x4
    block of the target frame, which is a stronger statement than "both were
    resized from the same source" and is the reason the target is built from
    these same delivered frames rather than re-decoded from the original.
    """
    import cv2

    if plane.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
        plane = cv2.resize(
            plane, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
    return reduce(plane)


def _duv_frame(
    scene_dir: Path, ordinal: int, *, driving: bool, inverted: bool
) -> np.ndarray:
    """One 336x192 DUV frame, composed from the arrays rather than resized.

    Never a resize of the delivered `duv.mp4`: its red channel is a log-depth
    code and its green and blue are a class palette, so interpolating it
    averages codes that mean nothing in between and paints classes nothing
    predicted. The median keeps a depth that occurred rather than averaging
    across a silhouette into a surface that does not exist, and the vote keeps
    whichever class owns the block.
    """
    metres = frames.read_array(scene_dir, "depth", ordinal).astype(np.float32)
    ids = frames.read_array(scene_dir, "semantic", ordinal).astype(np.uint8)
    return proxy.compose_proxy_frame(
        _to_duv_grid(metres, contract.downsample_depth),
        _to_duv_grid(ids, contract.downsample_semantic),
        driving=driving,
        inverted_depth=inverted,
    )


def _target_frames(
    scene_dir: Path, window: Window, *, source_video: Path | None
) -> list[np.ndarray]:
    """The window's colour frames at 1344x768, from wherever they were asked for."""
    if source_video is None:
        return [
            _resize(frames.read_image(scene_dir, "color", ordinal), TARGET_WIDTH, TARGET_HEIGHT)
            for ordinal in window.ordinals
        ]

    # Decoded straight to the target size, so the source's own 1920x1080 is
    # reduced once rather than reduced to 1280x720 and grown back.
    from .video import iter_frames

    wanted = set(window.ordinals)
    picked: dict[int, np.ndarray] = {}
    ordinal = 0
    for batch in iter_frames(source_video, size=(TARGET_WIDTH, TARGET_HEIGHT), chunk=64):
        for frame in batch:
            if ordinal in wanted:
                picked[ordinal] = frame
            ordinal += 1
        if ordinal > window.ordinals[-1]:
            break
    missing = [o for o in window.ordinals if o not in picked]
    if missing:
        raise ClipError(
            f"{source_video} ran out at frame {ordinal}, short of {missing[0]}; the "
            "delivered segment and the source video disagree about the episode's length"
        )
    return [picked[ordinal] for ordinal in window.ordinals]


# ---------------------------------------------------------------- annotations


def _slice_actions(payload, ordinals: tuple[int, ...]):
    """Keep the per-frame entries this window covers, in output order.

    The shape is whatever the corpus ships - a bare list, or a dict with the
    list under one of a few keys - so the wrapper is preserved and only the
    list inside it is cut. Anything that is not as long as the episode is left
    alone: guessing that a shorter array is also per-frame, and slicing it by
    frame index, would silently mispair actions with pictures.
    """
    if isinstance(payload, list):
        return [payload[o] for o in ordinals if o < len(payload)]
    if isinstance(payload, dict):
        out = dict(payload)
        for key, value in payload.items():
            if isinstance(value, list) and len(value) > ordinals[-1]:
                out[key] = [value[o] for o in ordinals]
        return out
    return payload


def _write_annotations(
    annotations: Path | None, window: Window, clip_dir: Path, report: dict
) -> dict:
    """Cut the episode's annotations down to this window.

    Split by what the claim is about. `action.json` is per frame, so it is cut.
    `caption.json` describes the episode, and an episode's caption is still
    true of five seconds of it, so it is copied whole and marked as such. The
    COLMAP model is neither: it is a reconstruction of the whole walk, sized in
    megabytes, and rewriting its text into a per-clip model would make this
    pipeline a second source of truth for geometry it did not solve. What the
    clip gets instead is the pose track for its own frames, in this package's
    own format, clearly derived - and the report says where the original is.
    """
    from .datasets import abot

    if annotations is None or not Path(annotations).is_file():
        return {"source": None}

    out = clip_dir / ANNOTATION_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    members = abot.read_members(Path(annotations))
    written: dict = {"source": str(annotations), "caption": None, "actions": None}

    if "action.json" in members:
        actions = _slice_actions(json.loads(members["action.json"]), window.ordinals)
        (out / "action.json").write_text(json.dumps(actions))
        written["actions"] = len(actions) if isinstance(actions, list) else "wrapped"
    if "caption.json" in members:
        (out / "caption.json").write_bytes(members["caption.json"])
        written["caption"] = "episode-level, copied whole"

    written["cameras"] = _write_cameras(members, window, out)
    return written


def _write_cameras(members: dict, window: Window, out: Path) -> str | None:
    """This window's COLMAP poses as an npz, or None with the reason recorded.

    Registration is not guaranteed to be complete - a sparse reconstruction
    drops frames it cannot solve - so the ordinals actually present are written
    alongside the poses rather than assumed to be the window's own.
    """
    from . import cameras as camera_io

    if "cameras.txt" not in members or "images.txt" not in members:
        return None

    with tempfile.TemporaryDirectory() as scratch:
        sparse = Path(scratch)
        for name in abot_sparse_members(members):
            (sparse / name).write_bytes(members[name])
        try:
            track = camera_io.from_colmap_text(sparse)
            names = camera_io.colmap_registered_names(sparse)
        except (ValueError, KeyError, OSError):
            return None

    ordinals = []
    for name in names:
        stem = Path(name).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        ordinals.append(int(digits) if digits else -1)
    ordinals = np.asarray(ordinals, dtype=np.int64)

    wanted = np.asarray(window.ordinals, dtype=np.int64)
    keep = np.flatnonzero(np.isin(ordinals, wanted))
    if keep.size == 0:
        return None

    subset = track.subset(keep)
    np.savez(
        out / "cameras.npz",
        cam2world=subset.cam2world,
        intrinsics=subset.intrinsics,
        # Never metric: a sparse reconstruction pins the scene only up to a
        # similarity, so these poses cannot be mixed with the metric depth
        # without a scale that this pipeline does not have.
        metric=np.array(False),
        source_ordinals=ordinals[keep],
    )
    return f"{keep.size} of {len(window.ordinals)} frames registered"


def abot_sparse_members(members: dict) -> list[str]:
    from .datasets.abot import SPARSE_MEMBERS

    return [name for name in SPARSE_MEMBERS if name in members]


# ----------------------------------------------------------------- one clip


def already_cut(clip_dir: Path, length: int = CLIP_FRAMES) -> bool:
    """Whether a previous run left a complete clip here.

    Frame counts rather than existence, for the same reason `delivery` checks
    them: a run killed mid-encode leaves two openable files that are short.
    """
    clip_dir = Path(clip_dir)
    videos = (
        clip_dir / TARGET_DIRNAME / TARGET_NAME,
        clip_dir / PROXY_DIRNAME / DUV_NAME,
    )
    if not all(path.is_file() and path.stat().st_size for path in videos):
        return False
    if not (clip_dir / TARGET_DIRNAME / ANCHOR_NAME).is_file():
        return False
    try:
        return all(probe(path).frames == length for path in videos)
    except (OSError, ValueError):
        return False


def cut_clip(
    scene_dir: Path,
    window: Window,
    clip_dir: Path,
    *,
    annotations: Path | None = None,
    source_video: Path | None = None,
    color_crf: int = proxy.DEFAULT_COLOR_CRF,
    fps: float = CLIP_FPS,
) -> dict:
    """Write one clip: target video, anchor, DUV, annotations, report."""
    import cv2

    scene_dir, clip_dir = Path(scene_dir), Path(clip_dir)
    report = _read_scene_report(scene_dir)
    driving = bool(
        report.get("semantic", {}).get("hero_split", {}).get("driving", False)
    )
    inverted = bool(report.get("duv_depth_inverted", False))

    (clip_dir / TARGET_DIRNAME).mkdir(parents=True, exist_ok=True)
    (clip_dir / PROXY_DIRNAME).mkdir(parents=True, exist_ok=True)

    targets = _target_frames(scene_dir, window, source_video=source_video)
    anchor = targets[0]

    rgb = proxy.open_encoder(
        clip_dir / TARGET_DIRNAME / TARGET_NAME,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        fps,
        kind="color",
        crf=color_crf,
    )
    duv = proxy.open_encoder(
        clip_dir / PROXY_DIRNAME / DUV_NAME, DUV_WIDTH, DUV_HEIGHT, fps, kind="proxy"
    )
    try:
        for frame, ordinal in zip(targets, window.ordinals):
            rgb.write(frame)
            duv.write(_duv_frame(scene_dir, ordinal, driving=driving, inverted=inverted))
    finally:
        rgb.close()
        duv.close()

    ok, buffer = cv2.imencode(".png", anchor[:, :, ::-1])
    if not ok:
        raise ClipError(f"failed to PNG-encode the anchor for {clip_dir.name}")
    (clip_dir / TARGET_DIRNAME / ANCHOR_NAME).write_bytes(buffer.tobytes())

    written = {
        "clip": clip_dir.name,
        "scene": window.scene,
        "sample_id": report.get("scene"),
        "source_video": report.get("source_video"),
        "window": window.index,
        # Every source frame this clip is made of, so the mapping back into the
        # episode never has to be re-derived from a rate and a start.
        "source_ordinals": list(window.ordinals),
        "source_fps": report.get("fps"),
        "frames": len(window.ordinals),
        "fps": fps,
        "target_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "target_from": "source video" if source_video else "delivered frames",
        "duv_size": [DUV_WIDTH, DUV_HEIGHT],
        "duv_depth_inverted": inverted,
        "taxonomy": report.get("config", {}).get("semantic_backend"),
        "deliverable": report.get("deliverable", True),
        "annotations": _write_annotations(annotations, window, clip_dir, report),
    }
    _atomic_json(clip_dir / CLIP_REPORT_NAME, written)
    return written


def _read_scene_report(scene_dir: Path) -> dict:
    from .delivery import REPORT_NAME

    path = Path(scene_dir) / REPORT_NAME
    if not path.is_file():
        raise ClipError(
            f"{scene_dir} has no {REPORT_NAME}; only a completed segment can be cut, "
            "and the DUV needs the settings it was delivered with"
        )
    return json.loads(path.read_text())


def _atomic_json(path: Path, payload: dict) -> None:
    handle, scratch = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    tmp = Path(scratch)
    try:
        with os.fdopen(handle, "w") as file:
            json.dump(payload, file, indent=2)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------- driving


def windows_for_scene(
    scene_dir: Path,
    *,
    count: int = CLIPS_PER_SCENE,
    length: int = CLIP_FRAMES,
    fps: float = CLIP_FPS,
) -> list[Window]:
    """Plan one segment's windows from what it recorded about itself."""
    report = _read_scene_report(scene_dir)
    return plan_windows(
        Path(scene_dir).name,
        int(report["frames"]),
        source_fps=float(report["fps"]),
        count=count,
        length=length,
        fps=fps,
    )


def cut_scene(
    scene_dir: Path,
    clips_root: Path,
    *,
    count: int = CLIPS_PER_SCENE,
    length: int = CLIP_FRAMES,
    fps: float = CLIP_FPS,
    color_crf: int = proxy.DEFAULT_COLOR_CRF,
    target_from_source: bool = False,
    resume: bool = False,
    progress=None,
) -> list[dict]:
    """Cut every window of one delivered segment."""
    scene_dir, clips_root = Path(scene_dir), Path(clips_root)
    report = _read_scene_report(scene_dir)
    annotations = scene_dir / "annotations.tar"
    source = Path(report["source_video"]) if target_from_source else None

    reports = []
    for window in windows_for_scene(scene_dir, count=count, length=length, fps=fps):
        clip_dir = clips_root / clip_name(window.scene, window.index)
        if resume and already_cut(clip_dir, length):
            existing = clip_dir / CLIP_REPORT_NAME
            reports.append(
                json.loads(existing.read_text())
                if existing.is_file()
                else {"clip": clip_dir.name, "skipped": "already cut"}
            )
            continue
        if progress is not None:
            progress(f"{clip_dir.name} <- {window.scene} frames {window.start}..{window.stop}")
        reports.append(
            cut_clip(
                scene_dir,
                window,
                clip_dir,
                annotations=annotations if annotations.is_file() else None,
                source_video=source,
                color_crf=color_crf,
                fps=fps,
            )
        )
    return reports


def write_clips_manifest(
    clips_root: Path, reports: list[dict], *, name: str | None = None
) -> Path:
    """One line per clip, saying which episode frames it holds."""
    clips_root = Path(clips_root)
    clips_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "clips": [
            {
                "clip": item.get("clip"),
                "scene": item.get("scene"),
                "window": item.get("window"),
                "source_video": item.get("source_video"),
                "source_ordinals": [
                    item.get("source_ordinals", [None])[0],
                    item.get("source_ordinals", [None])[-1],
                ]
                if item.get("source_ordinals")
                else None,
                "frames": item.get("frames"),
            }
            for item in reports
        ],
        "count": len(reports),
        "frames_each": CLIP_FRAMES,
        "fps": CLIP_FPS,
    }
    path = clips_root / (name or CLIPS_MANIFEST_NAME)
    _atomic_json(path, payload)
    return path


def audit_clips(clips_root: Path, length: int = CLIP_FRAMES) -> dict:
    """Count complete and short clips under a root, by opening their videos."""
    clips_root = Path(clips_root)
    complete, incomplete = [], []
    for clip_dir in sorted(clips_root.glob(f"{CLIP_PREFIX}*")):
        if not clip_dir.is_dir():
            continue
        (complete if already_cut(clip_dir, length) else incomplete).append(clip_dir.name)
    return {
        "clips_root": str(clips_root),
        "complete": len(complete),
        "incomplete": len(incomplete),
        "incomplete_clips": incomplete[:20],
        "frames_each": length,
    }


def expected_total(scenes: int, count: int = CLIPS_PER_SCENE) -> int:
    return scenes * count


def seconds_per_clip(length: int = CLIP_FRAMES, fps: float = CLIP_FPS) -> float:
    """Handy in messages, and a reminder that 124 at 24 is not five seconds."""
    return math.floor(length / fps * 100) / 100
