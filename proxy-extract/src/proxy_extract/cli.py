"""Command line entry point for the extraction and scoring commands.

Two groups. `qc`, `camera-qc`, `extract`, `validate` and `preview` build and
check a condition_root. `gtaweb-probe` and `player-bench` score against
gta-web's engine ground truth and need neither a GPU nor a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cameras as camera_io
from . import contract
from .pipeline import ExtractionConfig, condition_dir_for, extract_clip, extract_dataset, shard
from .proxy import DEFAULT_COLOR_CRF
from .qc import score_dataset, score_pair, write_report

DEPTH_BACKENDS = ("mapanything", "depth_anything", "synthetic")
SEMANTIC_BACKENDS = ("ade20k", "cityscapes", "coarse6", "standard11", "synthetic")
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
        help="write DATA_F.md delivery segments (1280x720 color/depth/semantic/duv + annotation)",
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
    scenes.add_argument("--out", type=Path, required=True, help="root to hold seg_long_NNNNNN/")
    scenes.add_argument("--depth-backend", choices=DEPTH_BACKENDS, default="mapanything")
    scenes.add_argument("--semantic-backend", choices=SEMANTIC_BACKENDS, default="standard11")
    scenes.add_argument("--refiner", choices=REFINERS, default="none")
    scenes.add_argument(
        "--chunk-frames", type=int, default=64, metavar="N",
        help="frames per model call; bounds GPU activation memory, not the host stacks",
    )
    scenes.add_argument("--temporal-radius", type=int, default=2)
    scenes.add_argument("--no-flow-compensate", action="store_true")
    scenes.add_argument(
        "--flow-downscale", type=int, default=2, metavar="N",
        help="solve optical flow at 1/N of the delivery size; 1 disables the shortcut",
    )
    scenes.add_argument(
        "--color-crf", type=int, default=None,
        help=f"x264 quality for high_color.mp4 (default {DEFAULT_COLOR_CRF}); 0 is lossless",
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
    scene_preview.add_argument("--scene", type=Path, required=True, help="a seg_long_NNNNNN dir")
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

    probe = sub.add_parser(
        "gtaweb-probe",
        help="check that a gta-web clip decodes the way DATA_F.md says it does",
    )
    probe.add_argument("--clips-dir", type=Path, required=True)
    probe.add_argument("--limit", type=int, default=3, help="how many clips to sample")

    bench = sub.add_parser(
        "player-bench",
        help="score the player/ped split against gta-web's own labels (CPU only)",
    )
    bench.add_argument("--clips-dir", type=Path, required=True)
    bench.add_argument("--limit", type=int, help="stop after this many clips")
    bench.add_argument("--report", type=Path)
    bench.add_argument(
        "--sweep",
        action="store_true",
        help="grid the screen anchor instead of scoring the default once",
    )

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


def _run_scenes(args: argparse.Namespace) -> int:
    from . import delivery

    videos = resolve_videos(args.video, recursive=args.recursive)
    assignments = delivery.assign_scenes(delivery.episodes_from_videos(videos))

    config = delivery.DeliveryConfig(
        depth_backend=args.depth_backend,
        semantic_backend=args.semantic_backend,
        semantic_refiner=args.refiner,
        chunk_frames=args.chunk_frames,
        temporal_radius=args.temporal_radius,
        flow_compensate=not args.no_flow_compensate,
        flow_downscale=args.flow_downscale,
        split_hero=not args.no_hero_split,
        inverted_duv_depth=args.inverted_duv_depth,
        fps=args.fps,
        **({"color_crf": args.color_crf} if args.color_crf is not None else {}),
    )

    # Written from the full assignment list before sharding, so every worker
    # records the same scene numbering rather than only its own slice.
    manifest = delivery.write_manifest(args.out, assignments)

    mine = assignments
    if args.shard:
        index, _, count = args.shard.partition("/")
        mine = shard(assignments, int(index), int(count))
        print(f"shard {args.shard}: {len(mine)} of {len(assignments)} episodes")
        if not mine:
            return 0
    else:
        print(f"{len(assignments)} episodes -> {args.out}, manifest at {manifest}")

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


def _gtaweb_clips(clips_dir: Path, limit: int | None):
    """Pair each `clips.json` entry with its semantic file, skipping incomplete ones."""
    from .datasets import gtaweb

    entries = gtaweb.load_clips(clips_dir)
    for clip in entries[:limit] if limit else entries:
        if clip.semantic and (clips_dir / clip.semantic).exists():
            yield clip


def _run_gtaweb_probe(args: argparse.Namespace) -> int:
    """Answer "does this corpus read the way the docs claim" before a batch run."""
    from .datasets import gtaweb

    clips = list(_gtaweb_clips(args.clips_dir, args.limit))
    if not clips:
        raise SystemExit(f"no clips with a semantic file under {args.clips_dir}")

    failures = 0
    for clip in clips:
        depth_path = args.clips_dir / clip.depth if clip.depth else None
        report = gtaweb.probe_decode(depth_path, args.clips_dir / clip.semantic)
        print(f"\n{clip.semantic}  tag={clip.tag}")
        for stream, result in report.items():
            if result.get("ok"):
                print(f"  {stream:9s} ok  " + json.dumps({k: v for k, v in result.items() if k != "ok"}))
            else:
                failures += 1
                print(f"  {stream:9s} FAILED  {result['error']}")

        if depth_path is not None and report.get("depth", {}).get("ok"):
            depth = gtaweb.decode_depth(depth_path, limit=8)
            suspicion = gtaweb.lossy_depth_suspicion(depth)
            verdict = "lossless" if suspicion["lossless"] else "LOSSY — values were interpolated"
            print(f"  depth compression: {verdict} ({suspicion['distinct_values']} distinct values)")

    print(f"\n{len(clips)} clips probed, {failures} stream(s) failed")
    return 1 if failures else 0


def _run_player_bench(args: argparse.Namespace) -> int:
    from .benchmark import player_bench
    from .datasets import gtaweb

    clips = list(_gtaweb_clips(args.clips_dir, args.limit))
    if not clips:
        raise SystemExit(f"no clips with a semantic file under {args.clips_dir}")

    loaded = []
    for clip in clips:
        try:
            loaded.append((gtaweb.decode_semantic(args.clips_dir / clip.semantic), clip.tag))
        except gtaweb.DecodeError as error:
            print(f"{clip.semantic}: {error}")

    if not loaded:
        raise SystemExit("no clip decoded; run `gtaweb-probe` to see why")

    if args.sweep:
        anchors = [(x / 20, y / 20) for x in range(8, 13) for y in range(9, 14)]
        rows = player_bench.sweep_anchor(loaded, anchors)
        rows.sort(key=lambda row: row["accuracy"], reverse=True)
        print(f"{'anchor':>14s} {'accuracy':>9s} {'decided':>8s}")
        for row in rows:
            print(
                f"  ({row['anchor'][0]:.2f}, {row['anchor'][1]:.2f}) "
                f"{row['accuracy']:9.3f} {row['decided_fraction']:8.3f}"
            )
        payload: dict = {"sweep": rows}
    else:
        measured = player_bench.measure_anchor([truth for truth, _ in loaded])
        scores = [player_bench.score_clip(truth, tag=tag) for truth, tag in loaded]
        summary = player_bench.aggregate(scores)

        print("\nwhere the protagonist actually sits:")
        print(json.dumps(measured, indent=2))
        print("\nsplit accuracy:")
        print(json.dumps(summary, indent=2))
        payload = {"measured_anchor": measured, **summary}

    if args.report:
        args.report.write_text(json.dumps(payload, indent=2))
        print(f"\nreport written to {args.report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "qc": _run_qc,
        "camera-qc": _run_camera_qc,
        "extract": _run_extract,
        "scenes": _run_scenes,
        "scenes-audit": _run_scenes_audit,
        "validate": _run_validate,
        "preview": _run_preview,
        "scenes-preview": _run_scenes_preview,
        "videos": _run_videos,
        "gtaweb-probe": _run_gtaweb_probe,
        "player-bench": _run_player_bench,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
