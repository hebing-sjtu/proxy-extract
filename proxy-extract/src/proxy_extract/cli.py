"""Command line entry point for the extraction commands.

Two deliverables, two groups. `scenes` and `scenes-audit` write and check the
1280x720 segments DATA_F.md specifies, which is what a delivery run produces.
`extract`, `videos`, `validate` and `preview` build and check the 336x192
`condition_root` that code-world-model consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import accel
from . import cameras as camera_io
from . import contract
from .frames import STREAMS as FRAME_STREAMS
from .pipeline import ExtractionConfig, condition_dir_for, extract_clip, extract_dataset, shard
from .proxy import DEFAULT_COLOR_CRF
from .streaming import DEFAULT_BLOCK
from .temporal import DEFAULT_MIN_RUN

DEPTH_BACKENDS = ("mapanything", "depth_anything", "depth_anything_v3", "synthetic")
SEMANTIC_BACKENDS = ("ade20k", "cityscapes", "coarse6", "standard11", "synthetic")
REFINERS = ("none", "sam3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proxy-extract", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="write a condition_root for one or more clips")
    extract.add_argument(
        "--video", type=Path, nargs="+", required=True,
        help="clip files, or directories to take every .mp4 from",
    )
    extract.add_argument(
        "--recursive",
        action="store_true",
        help="search directories recursively, for datasets that nest one dir per episode",
    )
    extract.add_argument(
        "--chunk-frames",
        type=int,
        metavar="N",
        help="run the models N frames at a time instead of whole-clip; needed for "
        "episodes much longer than the 124-frame window, and incompatible with --cameras",
    )
    extract.add_argument(
        "--emit-videos",
        choices=("none", "proxy", "all"),
        default="none",
        help="also write the DATA_F.md delivery videos beside each condition_root",
    )
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--depth-backend", choices=DEPTH_BACKENDS, default="mapanything")
    extract.add_argument("--semantic-backend", choices=SEMANTIC_BACKENDS, default="ade20k")
    extract.add_argument("--refiner", choices=REFINERS, default="none")
    extract.add_argument("--cameras", type=Path, help="GT camera track (.json or .npz), single clip only")
    extract.add_argument("--temporal-radius", type=int, default=2)
    extract.add_argument("--no-flow-compensate", action="store_true")
    extract.add_argument(
        "--depth-downsample", choices=("median", "min", "mean"), default="median"
    )
    extract.add_argument(
        "--shard",
        metavar="INDEX/COUNT",
        help="process only this worker's slice, e.g. --shard 3/8 for GPU 3 of 8",
    )
    extract.add_argument(
        "--resume", action="store_true", help="skip clips whose condition_root already validates"
    )
    extract.add_argument(
        "--keep-going", action="store_true", help="log and continue when a clip fails"
    )
    extract.add_argument(
        "--no-hero-split",
        action="store_true",
        help="with coarse6 or standard11, leave every person as npc/ped",
    )

    scenes = sub.add_parser(
        "scenes",
        help="write DATA_F.md delivery segments: per-frame frames/ plus the four "
        "1280x720 videos derived from them, and the source annotations",
    )
    scenes.add_argument(
        "--video", type=Path, nargs="+", required=True,
        help="episode files, or directories to take every .mp4 from",
    )
    scenes.add_argument(
        "--recursive",
        action="store_true",
        help="search directories recursively, for datasets that nest one dir per episode",
    )
    scenes.add_argument("--out", type=Path, required=True, help="root to hold seg_NNNNNN/")
    scenes.add_argument("--depth-backend", choices=DEPTH_BACKENDS, default="mapanything")
    scenes.add_argument(
        "--depth-backend-option", action="append", default=[], metavar="KEY=VALUE",
        help="pass a keyword to the depth backend's constructor, repeatable; "
        "e.g. --depth-backend-option window=4 for DA3's multi-view mode",
    )
    scenes.add_argument("--semantic-backend", choices=SEMANTIC_BACKENDS, default="standard11")
    scenes.add_argument(
        "--semantic-backend-option", action="append", default=[], metavar="KEY=VALUE",
        help="pass a keyword to the semantic backend's constructor, repeatable; "
        "e.g. --semantic-backend-option batch_size=16 to fill a larger GPU",
    )
    scenes.add_argument("--refiner", choices=REFINERS, default="none")
    scenes.add_argument(
        "--chunk-frames", type=int, default=64, metavar="N",
        help="frames decoded and handed to the models at a time; bounds GPU activation "
        "memory. The host side streams, so this is a throughput knob, not a memory ceiling",
    )
    scenes.add_argument(
        "--stabilise-block", type=int, default=DEFAULT_BLOCK, metavar="N",
        help=f"frames released per flow pass (default {DEFAULT_BLOCK}); larger repeats "
        "slightly less optical flow at the window seams and holds more frames while doing it",
    )
    scenes.add_argument(
        "--writer-threads", type=int, default=4, metavar="N",
        help="threads writing per-frame files, so the models are not waiting on a disk; "
        "0 writes inline",
    )
    scenes.add_argument(
        "--keep-frames", default=",".join(FRAME_STREAMS), metavar="STREAMS",
        help="which per-frame directories survive the encode, comma separated, or 'none' "
        f"(default {','.join(FRAME_STREAMS)}). Only depth carries what the videos cannot: "
        "depth.mp4 quantises float16 metres onto 8 bits. duv is derivable from depth and "
        "semantic, and semantic.mp4 already holds the same ids losslessly",
    )
    scenes.add_argument("--temporal-radius", type=int, default=2)
    scenes.add_argument(
        "--temporal-min-run", type=int, default=DEFAULT_MIN_RUN, metavar="N",
        help=f"erase label runs shorter than N frames (default {DEFAULT_MIN_RUN})",
    )
    scenes.add_argument("--no-flow-compensate", action="store_true")
    scenes.add_argument(
        "--flow-downscale", type=int, default=2, metavar="N",
        help="solve optical flow at 1/N of the delivery size; 1 disables the shortcut",
    )
    scenes.add_argument(
        "--color-crf", type=int, default=None,
        help=f"x264 quality for proxy/color.mp4 (default {DEFAULT_COLOR_CRF}); 0 is lossless",
    )
    scenes.add_argument(
        "--inverted-duv-depth", action="store_true",
        help="write the duv R channel with near as the high code; the default is forward "
        "(near 0, far 254, sky 255), which is what keeps the sky sentinel unambiguous",
    )
    scenes.add_argument("--fps", type=float, help="defaults to the source episode's rate")
    scenes.add_argument(
        "--no-hero-split", action="store_true",
        help="with coarse6 or standard11, leave every person as npc/ped",
    )
    scenes.add_argument(
        "--shard", metavar="INDEX/COUNT",
        help="process only this worker's slice, e.g. --shard 3/8 for GPU 3 of 8",
    )
    scenes.add_argument(
        "--resume", action="store_true", help="skip scenes whose videos already have every frame"
    )
    scenes.add_argument(
        "--quiet", action="store_true",
        help="do not report progress; a shard then says nothing until it finishes an episode",
    )
    scenes.add_argument(
        "--progress-interval", type=float, default=30.0, metavar="SECONDS",
        help="seconds between within-episode progress lines (default: %(default)s)",
    )
    scenes.add_argument(
        "--threads", type=int, default=None, metavar="N",
        help="cap OpenCV, torch and x264 to N threads each, so that workers sharing a "
        f"node do not each size their pools to the whole of it; defaults to ${accel.THREAD_VARIABLE}, "
        "which run_scenes.sh sets from the worker count",
    )
    scenes.add_argument(
        "--keep-going", action="store_true", help="log and continue when an episode fails"
    )

    audit = sub.add_parser(
        "scenes-audit",
        help="count complete/incomplete/missing scenes under an --out root",
    )
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--report", type=Path, help="also write the JSON here")

    validate = sub.add_parser("validate", help="re-read a condition_root and check it")
    validate.add_argument("--condition-root", type=Path, required=True)
    validate.add_argument("--expect-frames", type=int)

    preview = sub.add_parser("preview", help="render a condition_root to a viewable MP4")
    preview.add_argument("--condition-root", type=Path, required=True)
    preview.add_argument("--out", type=Path, required=True)
    preview.add_argument("--fps", type=int, default=24)

    scene_preview = sub.add_parser(
        "scenes-preview",
        help="render a delivered scene to a viewable contact sheet or MP4",
    )
    scene_preview.add_argument("--scene", type=Path, required=True, help="a seg_NNNNNN dir")
    scene_preview.add_argument(
        "--out", type=Path, required=True, help=".png for a contact sheet, else an MP4"
    )
    scene_preview.add_argument(
        "--frames", type=int, default=6, help="samples in a contact sheet; ignored for MP4"
    )
    scene_preview.add_argument("--fps", type=float, default=12.0)
    scene_preview.add_argument("--width", type=int, default=640, help="width of one panel")

    videos = sub.add_parser(
        "videos",
        help="encode a condition_root into the DATA_F.md depth/semantic/proxy videos",
    )
    videos.add_argument("--condition-root", type=Path, required=True)
    videos.add_argument("--out", type=Path, help="defaults to the condition_root itself")
    videos.add_argument("--fps", type=float, help="defaults to the source clip's rate, else 24")
    videos.add_argument(
        "--kinds",
        nargs="+",
        choices=("depth", "semantic", "proxy"),
        default=("depth", "semantic", "proxy"),
    )
    videos.add_argument(
        "--inverted-proxy-depth",
        action="store_true",
        help="write the proxy R channel with near as the high code; the default is forward "
        "(near 0, far 254, sky 255), which is what keeps the sky sentinel unambiguous",
    )

    return parser


VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")


def parse_backend_options(pairs: list[str]) -> dict:
    """Turn `KEY=VALUE` strings into constructor keywords.

    Values are read as Python literals where that works, so `window=4` arrives
    as an int rather than the string "4", and anything unquoted that is not a
    literal — a checkpoint name, say — stays a string.
    """
    import ast

    options: dict = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise SystemExit(f"backend option {pair!r} is not KEY=VALUE")
        try:
            options[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            options[key] = value
    return options


def parse_kept_streams(value: str) -> tuple[str, ...]:
    """Read `--keep-frames` into the stream names delivery expects.

    Named rather than numbered, and refused rather than ignored when unknown:
    getting this wrong deletes the expensive half of a scene, and a typo that
    silently kept nothing would not be noticed until the run was over.
    """
    text = value.strip().lower()
    if text in {"none", ""}:
        return ()
    if text == "all":
        return FRAME_STREAMS
    chosen = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in chosen if name not in FRAME_STREAMS]
    if unknown:
        raise SystemExit(
            f"--keep-frames does not know {', '.join(unknown)}; "
            f"expected some of {', '.join(FRAME_STREAMS)}, or none"
        )
    return chosen


def resolve_videos(paths: list[Path], *, recursive: bool = False) -> list[Path]:
    """Expand any directories in `--video` into the clips they hold.

    Shell globbing covers this locally, but not when the argument names a path
    inside a container that the launching shell cannot see, which is exactly
    how the sharded runs are started. Doing it here also keeps the ordering
    under our control: `shard` partitions by position, so every worker must
    derive the identical list or clips get processed twice or not at all.
    """
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            # Recursive for the datasets that nest one directory per episode:
            # ABot puts every clip at data/<prefix>/<sample_id>/video.mp4, so a
            # single-level listing of the root finds nothing at all.
            children = path.rglob("*") if recursive else path.iterdir()
            found = sorted(
                child
                for child in children
                if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES
            )
            if not found:
                where = "under" if recursive else "in"
                raise SystemExit(f"no {'/'.join(VIDEO_SUFFIXES)} files {where} {path}")
            resolved.extend(found)
        elif path.exists():
            resolved.append(path)
        else:
            raise SystemExit(f"no such video: {path}")
    return resolved


def _run_extract(args: argparse.Namespace) -> int:
    args.video = resolve_videos(args.video, recursive=args.recursive)
    if args.cameras and len(args.video) != 1:
        raise SystemExit("--cameras applies to a single --video; run per clip or use the Python API")

    config = ExtractionConfig(
        depth_backend=args.depth_backend,
        semantic_backend=args.semantic_backend,
        semantic_refiner=args.refiner,
        temporal_radius=args.temporal_radius,
        flow_compensate=not args.no_flow_compensate,
        depth_downsample=args.depth_downsample,
        split_hero=not args.no_hero_split,
        chunk_frames=args.chunk_frames,
    )

    videos = list(args.video)
    if args.shard:
        index, _, count = args.shard.partition("/")
        videos = shard(videos, int(index), int(count))
        print(f"shard {args.shard}: {len(videos)} of {len(args.video)} clips")
        if not videos:
            return 0

    if len(videos) == 1 and not args.shard:
        track = camera_io.load(args.cameras) if args.cameras else None
        reports = [
            extract_clip(
                videos[0],
                condition_dir_for(args.out, videos[0]),
                config=config,
                cameras=track,
            )
        ]
    else:
        reports = extract_dataset(
            videos,
            args.out,
            config=config,
            resume=args.resume,
            on_error="skip" if args.keep_going else "raise",
        )

    kinds = {"proxy": ("proxy",), "all": ("depth", "semantic", "proxy")}.get(args.emit_videos)

    for report in reports:
        if "skipped" in report or "failed" in report:
            print(f"{report['clip']}: {report.get('skipped') or report['failed']}")
            continue
        semantic, depth = report["semantic"], report["depth"]
        print(
            f"{report['clip']}: {report['frames']} frames, "
            f"depth {depth['metric_source']} median {depth['median_metres']} m, "
            f"flicker {semantic['flicker_before']:.4f} -> {semantic['flicker_after']:.4f}, "
            f"{report['elapsed_seconds']}s"
        )
        if kinds:
            from .proxy import write_videos

            summary = write_videos(Path(report["condition_root"]), kinds=kinds)
            print(f"  videos: {', '.join(sorted(summary['videos']))}")
    return 0


def _say(message: str) -> None:
    """One progress line, timestamped and flushed.

    Flushed because a shard's stdout is a log file, not a terminal, so Python
    block-buffers it: without this a worker that is running normally looks
    identical to one that is wedged for as long as it takes to fill 8 KB.
    """
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _run_scenes(args: argparse.Namespace) -> int:
    from . import delivery
    from .accel import limit_threads

    threads = limit_threads(args.threads)
    if threads and not args.quiet:
        _say(f"capped at {threads} CPU thread(s) per pool")

    videos = resolve_videos(args.video, recursive=args.recursive)
    assignments = delivery.assign_scenes(delivery.episodes_from_videos(videos))

    config = delivery.DeliveryConfig(
        depth_backend=args.depth_backend,
        depth_backend_options=parse_backend_options(args.depth_backend_option),
        semantic_backend=args.semantic_backend,
        semantic_backend_options=parse_backend_options(args.semantic_backend_option),
        semantic_refiner=args.refiner,
        chunk_frames=args.chunk_frames,
        stabilise_block=args.stabilise_block,
        writer_threads=args.writer_threads,
        keep_frames=parse_kept_streams(args.keep_frames),
        temporal_radius=args.temporal_radius,
        temporal_min_run=args.temporal_min_run,
        flow_compensate=not args.no_flow_compensate,
        flow_downscale=args.flow_downscale,
        split_hero=not args.no_hero_split,
        inverted_duv_depth=args.inverted_duv_depth,
        fps=args.fps,
        progress=None if args.quiet else _say,
        progress_interval=args.progress_interval,
        **({"color_crf": args.color_crf} if args.color_crf is not None else {}),
    )

    # Written from the full assignment list before sharding, so every worker
    # records the same scene numbering rather than only its own slice.
    manifest = delivery.write_manifest(args.out, assignments)

    mine = assignments
    if args.shard:
        index, _, count = args.shard.partition("/")
        mine = shard(assignments, int(index), int(count))
        _say(f"shard {args.shard}: {len(mine)} of {len(assignments)} episodes")
        if not mine:
            return 0
    else:
        _say(f"{len(assignments)} episodes -> {args.out}, manifest at {manifest}")

    reports = delivery.deliver_dataset(
        mine,
        args.out,
        config=config,
        resume=args.resume,
        on_error="skip" if args.keep_going else "raise",
    )

    failures = 0
    for report in reports:
        if "skipped" in report or "failed" in report:
            failures += "failed" in report
            print(f"{report['scene']}: {report.get('skipped') or report['failed']}")
            continue
        semantic, depth = report["semantic"], report["depth"]
        print(
            f"{report['scene']}: {report['frames']} frames @ {report['size'][0]}x{report['size'][1]}, "
            f"depth median {depth['median_metres']} m, "
            f"flicker {semantic['flicker_before']:.4f} -> {semantic['flicker_after']:.4f}, "
            f"{report['elapsed_seconds']}s"
        )
    return 1 if failures else 0


def _run_scenes_audit(args: argparse.Namespace) -> int:
    from . import delivery

    summary = delivery.audit(args.out)
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2))
        print(f"report written to {args.report}")
    # Non-zero while anything is outstanding, so a shell loop can wait on it.
    return 0 if summary["complete"] == summary["expected"] else 1


def _run_validate(args: argparse.Namespace) -> int:
    summary = contract.validate_condition_root(args.condition_root, expected_frames=args.expect_frames)
    print(json.dumps(summary, indent=2))
    return 0


def _run_preview(args: argparse.Namespace) -> int:
    from .preview import render_preview

    path = render_preview(args.condition_root, args.out, fps=args.fps)
    print(f"preview written to {path}")
    return 0


def _run_scenes_preview(args: argparse.Namespace) -> int:
    from .preview import render_scene_preview

    path = render_scene_preview(
        args.scene, args.out, frames=args.frames, fps=args.fps, width=args.width
    )
    print(f"preview written to {path}")
    return 0


def _run_videos(args: argparse.Namespace) -> int:
    from .proxy import write_videos

    summary = write_videos(
        args.condition_root,
        args.out,
        fps=args.fps,
        kinds=tuple(args.kinds),
        inverted_proxy_depth=args.inverted_proxy_depth,
    )
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "extract": _run_extract,
        "scenes": _run_scenes,
        "scenes-audit": _run_scenes_audit,
        "validate": _run_validate,
        "preview": _run_preview,
        "scenes-preview": _run_scenes_preview,
        "videos": _run_videos,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
