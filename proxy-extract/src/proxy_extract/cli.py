"""Command line entry point: `proxy-extract {qc,extract,validate,preview}`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cameras as camera_io
from . import contract
from .pipeline import ExtractionConfig, condition_dir_for, extract_clip, extract_dataset, shard
from .qc import score_dataset, score_pair, write_report

DEPTH_BACKENDS = ("mapanything", "depth_anything", "synthetic")
SEMANTIC_BACKENDS = ("ade20k", "cityscapes", "coarse6", "synthetic")
REFINERS = ("none", "sam3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proxy-extract", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    qc = sub.add_parser("qc", help="score high/low clip pairs for geometric drift")
    qc.add_argument("--dataset", type=Path, required=True, help="directory holding high/ and low/")
    qc.add_argument("--high-dir", default="high")
    qc.add_argument("--low-dir", default="low")
    qc.add_argument("--report", type=Path, help="write the per-clip JSON report here")

    camera_qc = sub.add_parser(
        "camera-qc",
        help="score each render against GT cameras (preferred over `qc` when tracks exist)",
    )
    camera_qc.add_argument("--dataset", type=Path, required=True)
    camera_qc.add_argument("--camera-dir", default="camera")
    camera_qc.add_argument(
        "--track", default="low", help="which render subdirectory to score, e.g. high or low"
    )
    camera_qc.add_argument("--report", type=Path)

    extract = sub.add_parser("extract", help="write a condition_root for one or more clips")
    extract.add_argument(
        "--video", type=Path, nargs="+", required=True,
        help="clip files, or directories to take every .mp4 from",
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
        help="with --semantic-backend coarse6, leave every person as npc",
    )

    validate = sub.add_parser("validate", help="re-read a condition_root and check it")
    validate.add_argument("--condition-root", type=Path, required=True)
    validate.add_argument("--expect-frames", type=int)

    preview = sub.add_parser("preview", help="render a condition_root to a viewable MP4")
    preview.add_argument("--condition-root", type=Path, required=True)
    preview.add_argument("--out", type=Path, required=True)
    preview.add_argument("--fps", type=int, default=24)

    return parser


def _run_qc(args: argparse.Namespace) -> int:
    reports = score_dataset(args.dataset, high_dir=args.high_dir, low_dir=args.low_dir)
    width = max(len(r.clip) for r in reports)
    header = f"{'clip':{width}s} {'tier':>7s} {'EPE/motion':>11s} {'flow cos':>9s}"
    print(header)
    print("-" * len(header))
    for report in sorted(reports, key=lambda r: r.epe_rel):
        print(f"{report.clip:{width}s} {report.tier:>7s} {report.epe_rel:11.3f} {report.flow_cos:9.3f}")

    summary = {tier: sum(1 for r in reports if r.tier == tier) for tier in ("keep", "review", "drop")}
    print(f"\n{len(reports)} clips: " + ", ".join(f"{v} {k}" for k, v in summary.items()))
    if args.report:
        write_report(reports, args.report)
        print(f"report written to {args.report}")
    return 0


def _run_camera_qc(args: argparse.Namespace) -> int:
    import numpy as np

    from .camera_qc import fidelity_tier, verify_track
    from .video import read_frames

    tracks = sorted((args.dataset / args.camera_dir).glob("*.json"))
    if not tracks:
        raise SystemExit(f"no camera tracks under {args.dataset / args.camera_dir}")

    rows = []
    for path in tracks:
        video_path = args.dataset / args.track / f"{path.stem}.mp4"
        if not video_path.exists():
            print(f"{path.stem}: no {args.track} render, skipped")
            continue

        track = camera_io.load(path)
        frames = read_frames(video_path, grayscale=True)
        # The delivered intrinsics target the 1280x720 high render; retarget
        # them if this render was delivered at another size.
        height, width = frames[0].shape
        if (width, height) != (1280, 720):
            focal = track.intrinsics[0, 0] * height / 720.0
            retargeted = np.array(
                [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
            )
            track = camera_io.CameraTrack(track.cam2world, retargeted, metric=track.metric)

        verdict = verify_track(frames, track, path.stem)
        rows.append(verdict)
        print(
            f"{verdict.clip:32s} poses:{verdict.tier:<9s} fidelity:{fidelity_tier(verdict.sampson_median):<8s}"
            f" sampson {verdict.sampson_median:7.2f} px  inlier {verdict.inlier_fraction:5.2f}"
        )

    summary = {
        tier: sum(1 for r in rows if fidelity_tier(r.sampson_median) == tier)
        for tier in ("keep", "review", "drop")
    }
    print(f"\n{len(rows)} clips: " + ", ".join(f"{v} {k}" for k, v in summary.items()))
    if args.report:
        args.report.write_text(
            json.dumps(
                [
                    {
                        "clip": r.clip,
                        "sampson_median": r.sampson_median,
                        "inlier_fraction": r.inlier_fraction,
                        "pose_tier": r.tier,
                        "fidelity_tier": fidelity_tier(r.sampson_median),
                        "note": r.note,
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        print(f"report written to {args.report}")
    return 0


VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")


def resolve_videos(paths: list[Path]) -> list[Path]:
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
            found = sorted(
                child for child in path.iterdir() if child.suffix.lower() in VIDEO_SUFFIXES
            )
            if not found:
                raise SystemExit(f"no {'/'.join(VIDEO_SUFFIXES)} files in {path}")
            resolved.extend(found)
        elif path.exists():
            resolved.append(path)
        else:
            raise SystemExit(f"no such video: {path}")
    return resolved


def _run_extract(args: argparse.Namespace) -> int:
    args.video = resolve_videos(args.video)
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
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    summary = contract.validate_condition_root(args.condition_root, expected_frames=args.expect_frames)
    print(json.dumps(summary, indent=2))
    return 0


def _run_preview(args: argparse.Namespace) -> int:
    from .preview import render_preview

    path = render_preview(args.condition_root, args.out, fps=args.fps)
    print(f"preview written to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "qc": _run_qc,
        "camera-qc": _run_camera_qc,
        "extract": _run_extract,
        "validate": _run_validate,
        "preview": _run_preview,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
