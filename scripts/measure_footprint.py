"""Measure the two numbers `run_scenes.sh` sizes a run with.

`GIB_PER_WORKER` decides how many workers fit on a node and `MIB_PER_SCENE`
whether the output filesystem can hold 2000 episodes. Both are quoted in the
runbook as measured values, and both were measured here. The defaults in the
launcher come from a laptop with synthetic backends; re-run this on the target
node with the real ones before committing to a long run, because the model's
own allocator is part of the footprint and it is not part of the default.

    python scripts/measure_footprint.py --frames 1800
    python scripts/measure_footprint.py --frames 1800 \
        --depth-backend depth_anything_v3 --semantic-backend standard11

Peak RSS is read from `ru_maxrss` on a sampling thread rather than at the end,
because the peak is reached inside a stage and released before it returns.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy-extract" / "src"))

from proxy_extract import delivery, frames  # noqa: E402

# ru_maxrss is bytes on macOS and kilobytes everywhere else.
RSS_SCALE = 2**20 if sys.platform == "darwin" else 2**10

# What an ABot episode is, and so what both launcher constants are quoted for.
EPISODE_FRAMES = 1800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames",
        type=int,
        default=EPISODE_FRAMES,
        metavar="N",
        help="episode length; the ABot episodes are about %(default)s frames (default: %(default)s)",
    )
    parser.add_argument(
        "--source-size",
        default="1920x1080",
        metavar="WxH",
        help="size of the synthetic input clip, before delivery resizes it (default: %(default)s)",
    )
    parser.add_argument("--depth-backend", default="synthetic")
    parser.add_argument("--semantic-backend", default="synthetic")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=64,
        help="what a shard would run with (default: %(default)s)",
    )
    parser.add_argument(
        "--keep",
        default=None,
        metavar="DIR",
        help="keep the scene here instead of in a temporary directory",
    )
    return parser.parse_args()


def write_clip(path: Path, *, frames_wanted: int, width: int, height: int) -> None:
    """A clip with motion and grain, so the encoders are not given a gift.

    A constant field compresses to nothing and stabilises perfectly, which
    would understate both the space and the time by a wide margin.
    """
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"could not open {path} for writing; is OpenCV built with an mp4 encoder?")

    rng = np.random.default_rng(7)
    box = height // 3
    for index in range(frames_wanted):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[: height // 2, :, 2] = 200
        frame[height // 2 :, :, 1] = 150
        left = 200 + index % 300
        frame[box : box * 2, left : left + width // 2] = 210
        frame += rng.integers(0, 12, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class PeakWatcher:
    """Sample `ru_maxrss` until told to stop."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak = 0
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while self._running:
            self.peak = max(self.peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            time.sleep(self.interval)

    def __enter__(self) -> PeakWatcher:
        self._running = True
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._running = False
        self._thread.join()
        self.peak = max(self.peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def stream_sizes(scene: Path) -> dict[str, int]:
    """Bytes on disk per per-frame stream, plus the videos they encode to."""
    sizes: dict[str, int] = {}
    frames_dir = frames.frames_dir_for(scene)
    for stream in frames.STREAMS:
        directory = frames_dir / stream
        if directory.is_dir():
            sizes[f"frames/{stream}"] = sum(
                path.stat().st_size for path in directory.iterdir() if path.is_file()
            )
    proxy_dir = scene / delivery.PROXY_DIRNAME
    if proxy_dir.is_dir():
        sizes["proxy/*.mp4"] = sum(path.stat().st_size for path in proxy_dir.glob("*.mp4"))
    return sizes


def run(args: argparse.Namespace, work: Path) -> None:
    width, height = (int(part) for part in args.source_size.lower().split("x"))
    clip = work / "video.mp4"
    print(f"writing a {args.frames}-frame {width}x{height} clip ...", flush=True)
    write_clip(clip, frames_wanted=args.frames, width=width, height=height)

    scene = work / f"{delivery.SCENE_PREFIX}000000"
    config = delivery.DeliveryConfig(
        depth_backend=args.depth_backend,
        semantic_backend=args.semantic_backend,
        chunk_frames=args.chunk_frames,
    )

    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.time()
    with PeakWatcher() as watcher:
        report = delivery.extract_scene(clip, scene, config=config)
    elapsed = time.time() - started

    count = report["frames"]
    peak_gib = watcher.peak / RSS_SCALE / 2**10
    sizes = stream_sizes(scene)
    frame_bytes = sum(value for key, value in sizes.items() if key.startswith("frames/"))

    print()
    print(f"frames        {count}")
    print(f"elapsed       {elapsed:.1f} s  ({elapsed / count * 1000:.0f} ms/frame)")
    print(f"peak RSS      {peak_gib:.2f} GiB  (baseline {baseline / RSS_SCALE / 2**10:.2f} GiB)")
    print()
    for name, value in sizes.items():
        print(f"{name:<18} {value / 2**20:>8.0f} MiB  ({value / count / 2**20:.3f} MiB/frame)")
    print()
    print(f"what the launcher wants, for a {EPISODE_FRAMES}-frame episode:")
    # Space is per-frame by construction, so scaling it is arithmetic. Memory is
    # not: stage 1 is flat in the episode length and stage 2 is linear in it, so
    # a short run's peak is mostly the flat part and scaling it would overstate
    # the flat part and understate the slope at once.
    per_scene = (frame_bytes + sizes.get("proxy/*.mp4", 0)) / count * EPISODE_FRAMES / 2**20
    print(f"  MIB_PER_SCENE={per_scene:.0f}  # frames/ plus proxy/, scaled from {count} frames")
    if count >= EPISODE_FRAMES:
        print(f"  GIB_PER_WORKER={peak_gib:.0f}  # measured peak RSS")
    else:
        print(
            f"  GIB_PER_WORKER=?  # {peak_gib:.2f} GiB at {count} frames does not scale;"
            f" re-run with --frames {EPISODE_FRAMES}"
        )
    if args.depth_backend == "synthetic" or args.semantic_backend == "synthetic":
        print()
        print("note: a synthetic backend allocates nothing, so the RSS above is the")
        print("      pipeline's alone. Re-run with the real backends for the number")
        print("      a shard will actually take.")


def main() -> None:
    args = parse_args()
    if args.keep:
        destination = Path(args.keep)
        destination.mkdir(parents=True, exist_ok=True)
        run(args, destination)
        return
    with tempfile.TemporaryDirectory() as work:
        run(args, Path(work))


if __name__ == "__main__":
    main()
