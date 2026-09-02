"""Per-frame intermediates, written before anything is encoded.

An episode used to go from decode to four finished mp4s inside one function
call, holding the whole thing in host memory on the way. That had two costs. A
worker killed at frame 1700 of 1800 left nothing behind and started again from
zero, and the ~40 GiB it held meant only three workers fit per GPU, which is
not enough of them to keep an H200 busy.

So the pipeline lands here first:

    seg_000000/frames/
        color/000000.png        RGB, lossless
        depth/000000.npy        float16 metres, 0 = no depth
        semantic/000000.npy     uint8 class ids
        duv/000000.png          RGB, lossless

and the videos are encoded from these afterwards. Every frame is written
independently, so "how far did this episode get" is a question the directory
answers by itself and a restart resumes rather than repeats.

Depth and semantics are arrays rather than images because that is what they
are. The delivered depth.mp4 quantises metres onto 8 bits and semantic.mp4
packs ids into a blue channel; both are encodings chosen for a video container,
and reading them back costs precision that the array beside them still has.
float16 holds 0.1 m to 8000 m to about 0.05% - some 60x finer than the 8-bit
video derived from it - so it is the smaller file that loses nothing anyone can
use.

Colour and DUV are PNG because they are already 8-bit RGB, where PNG is
lossless and about a third the size of the raw bytes. Encoding colour.mp4 from
these PNGs rather than straight from the decoder keeps the guarantee that the
four streams describe the same pixels: PNG round-trips exactly, so the frames
the models saw are the frames that get encoded.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FRAMES_DIRNAME = "frames"

# Working state, hidden because it is this pipeline's business rather than the
# delivered set's. It holds the one stream that exists only between stages.
STAGE_DIRNAME = ".stage"

# The four delivered streams. The two that carry numbers keep their dtype; the
# two that carry pixels go through a lossless image codec.
STREAMS = ("color", "depth", "semantic", "duv")

# Labels as they leave the flow-compensated window: stabilised, but before run
# suppression and before the protagonist is split out of the person class. Both
# of those need to see every frame at once, so they cannot run inside the
# streaming pass, and their input has to survive a restart somewhere that is not
# `semantic` - otherwise a worker killed halfway through writing the final
# labels would come back to a directory holding half of each and no way to tell
# which frame is which.
STAGING_STREAM = "labels"

ARRAY_STREAMS = ("depth", "semantic", STAGING_STREAM)
IMAGE_STREAMS = ("color", "duv")
ALL_STREAMS = (*STREAMS, STAGING_STREAM)

DEPTH_DTYPE = np.float16
LABEL_DTYPE = np.uint8

_SUFFIX = {
    "color": ".png",
    "depth": ".npy",
    "semantic": ".npy",
    "duv": ".png",
    STAGING_STREAM: ".npy",
}
_PARENT = {stream: FRAMES_DIRNAME for stream in STREAMS} | {STAGING_STREAM: STAGE_DIRNAME}

# PNG's slowest levels buy a few percent for several times the CPU, and this
# runs 1800 times per episode on a thread the GPU is waiting behind.
PNG_COMPRESSION = 1

# Written to `name.part` and renamed into place. A worker killed mid-write
# otherwise leaves a truncated file that looks finished to anything counting
# filenames, and the count is exactly what resume trusts.
PARTIAL_SUFFIX = ".part"


def frames_dir_for(scene_dir: Path) -> Path:
    return Path(scene_dir) / FRAMES_DIRNAME


def stage_dir_for(scene_dir: Path) -> Path:
    return Path(scene_dir) / STAGE_DIRNAME


def stream_dir(scene_dir: Path, stream: str) -> Path:
    if stream not in _PARENT:
        raise ValueError(f"unknown stream {stream!r}; expected one of {sorted(_PARENT)}")
    return Path(scene_dir) / _PARENT[stream] / stream


def make_dirs(scene_dir: Path, streams: tuple[str, ...] = ALL_STREAMS) -> dict[str, Path]:
    dirs = {stream: stream_dir(scene_dir, stream) for stream in streams}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def frame_path(scene_dir: Path, stream: str, ordinal: int) -> Path:
    return stream_dir(scene_dir, stream) / f"{ordinal:06d}{_SUFFIX[stream]}"


def _place(path: Path, write) -> Path:
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    write(partial)
    os.replace(partial, path)
    return path


def write_array(scene_dir: Path, stream: str, ordinal: int, array: np.ndarray) -> Path:
    """Persist one depth or label frame."""
    if stream not in ARRAY_STREAMS:
        raise ValueError(f"{stream!r} is not an array stream; expected one of {list(ARRAY_STREAMS)}")
    dtype = DEPTH_DTYPE if stream == "depth" else LABEL_DTYPE
    payload = np.ascontiguousarray(array, dtype=dtype)

    def write(target: Path) -> None:
        # Through a handle rather than a name: `np.save` appends `.npy` to any
        # path that does not already end in it, which silently defeats writing
        # to a `.part` file and renaming.
        with open(target, "wb") as handle:
            np.save(handle, payload, allow_pickle=False)

    return _place(frame_path(scene_dir, stream, ordinal), write)


def write_image(scene_dir: Path, stream: str, ordinal: int, rgb: np.ndarray) -> Path:
    """Persist one RGB frame losslessly."""
    import cv2

    if stream not in IMAGE_STREAMS:
        raise ValueError(f"{stream!r} is not an image stream; expected one of {list(IMAGE_STREAMS)}")
    bgr = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)

    def write(target: Path) -> None:
        # cv2 picks its writer from the extension, and `.png.part` is not one it
        # knows, so name the format explicitly.
        ok, buffer = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        if not ok:
            raise OSError(f"failed to PNG-encode {stream} frame {ordinal}")
        target.write_bytes(buffer.tobytes())

    return _place(frame_path(scene_dir, stream, ordinal), write)


def read_array(scene_dir: Path, stream: str, ordinal: int) -> np.ndarray:
    """Read one depth or label frame back.

    Depth widens to float32 on the way out: it was stored narrow to halve the
    file, but everything downstream computes in float32 and mixing the two
    silently changes accumulation behaviour.
    """
    array = np.load(frame_path(scene_dir, stream, ordinal), allow_pickle=False)
    return array.astype(np.float32) if stream == "depth" else array


def read_image(scene_dir: Path, stream: str, ordinal: int) -> np.ndarray:
    import cv2

    path = frame_path(scene_dir, stream, ordinal)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def read_stack(scene_dir: Path, stream: str, count: int) -> np.ndarray:
    """Read `count` array frames back as one stack.

    Only the label streams are read this way. Depth is never stacked - a whole
    episode of it is the 6.6 GB this module exists to avoid - but the
    protagonist tracker genuinely needs every frame at once, and at a byte a
    pixel it can have them.
    """
    if count <= 0:
        raise ValueError(f"count must be >= 1, got {count}")
    first = read_array(scene_dir, stream, 0)
    stack = np.empty((count, *first.shape), dtype=first.dtype)
    stack[0] = first
    for ordinal in range(1, count):
        stack[ordinal] = read_array(scene_dir, stream, ordinal)
    return stack


def drop_stream(scene_dir: Path, stream: str) -> None:
    """Delete one stream outright."""
    import shutil

    shutil.rmtree(stream_dir(scene_dir, stream), ignore_errors=True)


def clear_stage(scene_dir: Path) -> None:
    """Delete the working directory, leaving only what is delivered."""
    import shutil

    shutil.rmtree(stage_dir_for(scene_dir), ignore_errors=True)


def keep_only(scene_dir: Path, streams: tuple[str, ...]) -> None:
    """Delete every per-frame stream except the named ones.

    Run once the videos are written, because until then every stream is either
    an input to the encode or a resume point for it.
    """
    import shutil

    unknown = set(streams) - set(STREAMS)
    if unknown:
        raise ValueError(f"unknown stream(s) {sorted(unknown)}; expected some of {list(STREAMS)}")
    for stream in STREAMS:
        if stream not in streams:
            drop_stream(scene_dir, stream)
    if not streams:
        shutil.rmtree(frames_dir_for(scene_dir), ignore_errors=True)


def contiguous_count(scene_dir: Path, stream: str) -> int:
    """How many frames are present from 0 upward, without a gap.

    A gap means a worker died with writes in flight, and the frames past it
    cannot be trusted to line up with the ones before. Counting to the first
    hole rather than counting files keeps resume from stitching two halves of
    different runs together.
    """
    directory = stream_dir(scene_dir, stream)
    if not directory.is_dir():
        return 0
    suffix = _SUFFIX[stream]
    present = {
        int(entry.name[: -len(suffix)])
        for entry in directory.iterdir()
        if entry.name.endswith(suffix) and entry.name[: -len(suffix)].isdigit()
    }
    count = 0
    while count in present:
        count += 1
    return count


def complete_through(scene_dir: Path, streams: tuple[str, ...]) -> int:
    """The frame count every named stream has reached."""
    return min((contiguous_count(scene_dir, stream) for stream in streams), default=0)


def discard_from(scene_dir: Path, stream: str, ordinal: int) -> int:
    """Delete frames at or past `ordinal`, plus any half-written ones.

    Resume re-derives a few frames of context before the point it stopped at,
    so whatever a dying worker left in that overlap has to go first - otherwise
    the second run's output would be interleaved with the first's.
    """
    directory = stream_dir(scene_dir, stream)
    if not directory.is_dir():
        return 0
    suffix = _SUFFIX[stream]
    removed = 0
    for entry in directory.iterdir():
        if entry.name.endswith(PARTIAL_SUFFIX):
            entry.unlink()
            removed += 1
            continue
        stem = entry.name[: -len(suffix)] if entry.name.endswith(suffix) else None
        if stem is not None and stem.isdigit() and int(stem) >= ordinal:
            entry.unlink()
            removed += 1
    return removed
