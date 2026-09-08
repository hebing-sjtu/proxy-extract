"""Command line entry point.

`captions` is the only command that spends money. The rest exist so that the
things you would otherwise want to re-run it for - a phrasing change, a
different training window, a look at what it did - do not require it.

    captions            caption clips with a VLM, write annotations/prompt.json
    captions-evidence   the DUV table and the exact prompt, with no model call
    captions-recompile  re-render the text from structure already on disk
    captions-slice      cut a caption down to a sub-window of its clip
    captions-export     project captions onto CWM user sentences, write prompt.txt
    captions-audit      count what is captioned, what failed, and what disagreed
    captions-show       print one caption's compiled text
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import cwm_export, layout, observe, render, timeline, verify
from . import evidence as evidence_mod
from .contract import Caption

AUDIT_NAME = "captions_audit.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clip-prompts",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("captions", help="caption clips with a VLM")
    run.add_argument("--clips", type=Path, required=True, help="a clips root, or one clip dir")
    run.add_argument("--limit", type=int, help="only the first N clips, in name order")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--model", default=observe.DEFAULT_MODEL)
    run.add_argument("--backend", default=observe.DEFAULT_BACKEND)
    run.add_argument("--env-dir", type=Path, help="extra directory to read .env files from")
    run.add_argument("--bin-seconds", type=float, default=timeline.DEFAULT_BIN_SECONDS)
    run.add_argument(
        "--min-tail-seconds",
        type=float,
        default=timeline.DEFAULT_MIN_TAIL_SECONDS,
        help="a final bin shorter than this joins the one before it",
    )
    run.add_argument("--sample-fps", type=float, default=observe.DEFAULT_SAMPLE_FPS)
    run.add_argument("--max-frames", type=int, default=observe.DEFAULT_MAX_FRAMES)
    run.add_argument("--repairs", type=int, default=observe.DEFAULT_REPAIRS)
    run.add_argument(
        "--sheet",
        choices=("auto", "always", "never"),
        default="auto",
        help="attach a labelled one-frame-per-bin sheet; auto means up to 16 bins",
    )
    run.add_argument("--keep-sheet", action="store_true", help="leave the sheet on disk")
    run.add_argument("--overwrite", action="store_true", help="recaption clips already done")
    run.add_argument("--keep-going", action="store_true", help="one bad clip costs one clip")
    run.add_argument("--report", type=Path, help="also write the run summary here")

    ev = sub.add_parser("captions-evidence", help="DUV table and prompt, no model call")
    ev.add_argument("--clip", type=Path, required=True)
    ev.add_argument("--bin-seconds", type=float, default=timeline.DEFAULT_BIN_SECONDS)
    ev.add_argument("--min-tail-seconds", type=float, default=timeline.DEFAULT_MIN_TAIL_SECONDS)
    ev.add_argument("--json", action="store_true", help="the table as JSON instead of the prompt")
    ev.add_argument("--sheet", type=Path, help="also write the contact sheet here")

    recompile = sub.add_parser("captions-recompile", help="re-render text from structure")
    recompile.add_argument("--clips", type=Path, required=True)
    recompile.add_argument("--limit", type=int)
    recompile.add_argument(
        "--reverify",
        action="store_true",
        help="also re-run the evidence checks, without re-measuring the DUV",
    )

    cut = sub.add_parser("captions-slice", help="cut a caption to a sub-window")
    cut.add_argument("--prompt", type=Path, required=True)
    cut.add_argument("--start", type=float, required=True)
    cut.add_argument("--stop", type=float, required=True)
    cut.add_argument("--out", type=Path, help="defaults to stdout")

    export = sub.add_parser(
        "captions-export",
        help="project captions onto CWM user sentences",
        description=(
            "Write the flat user sentence CWM feeds Qwen. Deterministic: it reads "
            "prompt.json and calls no model. See clip-prompts/CWM_TEXT_EXPORT.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export.add_argument("--clips", type=Path, required=True, help="a clips root, or one clip dir")
    export.add_argument("--style", choices=cwm_export.VARIANTS, default="window")
    export.add_argument(
        "--prose",
        choices=cwm_export.PROSE_STYLES,
        default="lean",
        help="which global sentence the window variant carries",
    )
    export.add_argument("--write-txt", action="store_true", help="write <clip>/prompt.txt")
    export.add_argument("--overwrite", action="store_true", help="replace an existing prompt.txt")
    export.add_argument(
        "--keep-failed",
        action="store_true",
        help="write prompt.txt even when the verifier found a contradiction",
    )
    export.add_argument("--limit", type=int, help="only the first N clips, in name order")
    export.add_argument("--report", type=Path)

    audit = sub.add_parser("captions-audit", help="count captioned, failed and disagreeing clips")
    audit.add_argument("--clips", type=Path, required=True)
    audit.add_argument("--report", type=Path)
    audit.add_argument("--list", choices=("captioned", "missing", "failed", "warned"))

    show = sub.add_parser("captions-show", help="print a caption's compiled text")
    show.add_argument("--prompt", type=Path, required=True)
    show.add_argument("--style", choices=("timed", "lean", "rich", "cwm"), default="timed")
    show.add_argument(
        "--prose",
        choices=cwm_export.PROSE_STYLES,
        default="lean",
        help="for --style cwm: which global sentence to carry",
    )
    show.add_argument(
        "--conditioning",
        action="store_true",
        help="also print what the control video means, resolved from its card",
    )

    return parser


def _say(message: str) -> None:
    print(message, flush=True)


def _grid_for(clip: layout.Clip, bin_seconds: float, min_tail_seconds: float) -> timeline.Grid:
    frames, fps = clip.shape()
    return timeline.plan(
        frames, fps, bin_seconds=bin_seconds, min_tail_seconds=min_tail_seconds
    )


def _window(clip: layout.Clip) -> dict:
    report = clip.report()
    ordinals = report.get("source_ordinals") or []
    return {
        "clip": clip.name,
        "episode": report.get("scene"),
        "window": report.get("window"),
        "t0": 0.0,
        "frames": report.get("frames"),
        "fps": report.get("fps"),
        "duration": round(float(report["frames"]) / float(report["fps"]), 3),
        "source_fps": report.get("source_fps"),
        "source_ordinals": [ordinals[0], ordinals[-1] + 1] if ordinals else None,
        "source_video": report.get("source_video"),
    }


def _wants_sheet(mode: str, grid: timeline.Grid) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    return grid.count <= 16


def _caption_one(clip: layout.Clip, client, args) -> dict:
    clip.check()
    if not clip.usable():
        return {"clip": clip.name, "status": "skipped", "reason": "not deliverable"}
    if clip.prompt.is_file() and not args.overwrite:
        return {"clip": clip.name, "status": "reused"}

    grid = _grid_for(clip, args.bin_seconds, args.min_tail_seconds)
    facts = evidence_mod.measure(clip.duv, grid, hero_resolved=clip.hero_resolved())

    sheet = None
    if _wants_sheet(args.sheet, grid):
        clip.annotations.mkdir(parents=True, exist_ok=True)
        sheet = evidence_mod.contact_sheet(clip.rgb, grid, clip.sheet)

    try:
        seen = observe.observe(
            client,
            grid=grid,
            video=clip.rgb,
            briefing=evidence_mod.briefing(facts),
            sheet=sheet,
            model=args.model,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            repairs=args.repairs,
        )
    finally:
        if sheet is not None and not args.keep_sheet and sheet.is_file():
            sheet.unlink()

    caption = seen.caption
    caption = Caption(
        grid=caption.grid,
        scene=caption.scene,
        entities=caption.entities,
        events=caption.events,
        window=_window(clip),
        evidence=facts,
        provenance=observe.provenance(
            model=args.model,
            backend=args.backend,
            attempts=seen.attempts,
            usage=seen.usage,
            quality=seen.quality,
        ),
    ).bind()
    caption = _finish(caption)
    caption.write(clip.prompt)

    checks = caption.checks
    return {
        "clip": clip.name,
        "status": "written",
        "bins": caption.grid.count,
        "events": len(caption.events),
        "attempts": seen.attempts,
        "score": caption.provenance.get("score"),
        "fail": len(checks.get("fail") or ()),
        "warn": len(checks.get("warn") or ()),
    }


def _finish(caption: Caption) -> Caption:
    """Attach the checks, the score and the compiled text, in that order."""
    from dataclasses import replace

    caption = replace(caption, checks=verify.check(caption))
    provenance = dict(caption.provenance)
    provenance["score"] = verify.score(caption)
    return replace(caption, provenance=provenance, compiled=render.compile_all(caption))


def _run_captions(args: argparse.Namespace) -> int:
    clips = layout.discover(args.clips, limit=args.limit)
    if not clips:
        _say(f"no clips under {args.clips}")
        return 1
    client = observe.build_client(args.backend, env_dir=args.env_dir)

    rows: list[dict] = []

    def work(clip: layout.Clip) -> dict:
        try:
            row = _caption_one(clip, client, args)
        except Exception as exc:
            if not args.keep_going:
                raise
            row = {"clip": clip.name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc(file=sys.stderr)
        _say(json.dumps(row, ensure_ascii=False))
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(work, clips))

    summary = _summarise(rows)
    _say(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if summary["failed"] == 0 else 1


def _summarise(rows: list[dict]) -> dict:
    counts: dict = {"clips": len(rows), "written": 0, "reused": 0, "skipped": 0, "failed": 0}
    for row in rows:
        key = row.get("status", "failed")
        counts[key] = counts.get(key, 0) + 1
    counts["with_fail"] = sum(1 for row in rows if row.get("fail"))
    counts["with_warn"] = sum(1 for row in rows if row.get("warn"))
    return counts


def _run_evidence(args: argparse.Namespace) -> int:
    from . import prompts

    clip = layout.clip_at(args.clip)
    clip.check()
    grid = _grid_for(clip, args.bin_seconds, args.min_tail_seconds)
    facts = evidence_mod.measure(clip.duv, grid, hero_resolved=clip.hero_resolved())
    if args.sheet:
        _say(str(evidence_mod.contact_sheet(clip.rgb, grid, args.sheet)))
    if args.json:
        _say(json.dumps({"timeline": grid.as_dict(), "evidence": facts}, ensure_ascii=False, indent=2))
    else:
        _say(prompts.instruction(grid, evidence_mod.briefing(facts), sheet=bool(args.sheet)))
    return 0


def _run_recompile(args: argparse.Namespace) -> int:
    from dataclasses import replace

    done = 0
    for clip in layout.discover(args.clips, limit=args.limit):
        if not clip.prompt.is_file():
            continue
        caption = Caption.read(clip.prompt)
        if args.reverify:
            caption = _finish(caption)
        else:
            caption = replace(caption, compiled=render.compile_all(caption))
        caption.write(clip.prompt)
        done += 1
    _say(json.dumps({"recompiled": done}, ensure_ascii=False))
    return 0


def _run_slice(args: argparse.Namespace) -> int:
    caption = Caption.read(args.prompt)
    cut = _finish(caption.slice_to(args.start, args.stop))
    if args.out:
        cut.write(args.out)
        _say(str(args.out))
    else:
        _say(json.dumps(cut.as_dict(), ensure_ascii=False, indent=2))
    return 0


def _export_one(clip: layout.Clip, args: argparse.Namespace) -> dict:
    from dataclasses import replace

    caption = Caption.read(clip.prompt)
    compiled = caption.compiled or render.compile_all(caption)
    caption = replace(caption, compiled=compiled)
    user = cwm_export.user_text(caption, variant=args.style, prose=args.prose)

    row = {
        "clip": clip.name,
        "variant": args.style,
        "system": cwm_export.system_id(caption),
        "chars": len(user),
    }
    if not args.write_txt:
        row["status"] = "compiled"
        return row
    if not cwm_export.should_write_txt(caption, keep_failed=args.keep_failed):
        row["status"] = "skipped"
        row["why"] = "the verifier found a contradiction; --keep-failed writes it anyway"
        return row
    if clip.prompt_txt.exists() and not args.overwrite:
        row["status"] = "skipped"
        row["why"] = "prompt.txt already exists; --overwrite replaces it"
        return row
    cwm_export.write_prompt_txt(clip.prompt_txt, user)
    row["status"] = "written"
    row["path"] = str(clip.prompt_txt)
    return row


def _run_export(args: argparse.Namespace) -> int:
    clips = layout.discover(args.clips, limit=args.limit)
    if not clips:
        _say(f"no clips under {args.clips}")
        return 1

    rows: list[dict] = []
    for clip in clips:
        if not clip.prompt.is_file():
            # No invented fallback caption: a clip with no prompt.json is
            # counted and left alone. Writing a plausible sentence here is how
            # a corpus quietly acquires text nobody checked.
            rows.append({"clip": clip.name, "status": "missing"})
            continue
        try:
            rows.append(_export_one(clip, args))
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the corpus
            rows.append(
                {"clip": clip.name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )

    summary: dict = {"clips": len(rows)}
    for row in rows:
        key = row.get("status", "failed")
        summary[key] = summary.get(key, 0) + 1
    _say(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("written"):
        _say(
            "text changed, so the text embeddings must be recomputed on the "
            "training side. The VAE latents are unaffected.",
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if not summary.get("failed") else 1


# A user sentence always opens with its window stamp. Matching the shape rather
# than `[0.00s-` specifically so that a sliced continuation still counts.
EXPORTED_RE = re.compile(r"^\[\d+\.\d{2}s-\d+\.\d{2}s\] ")


def _is_exported(clip: layout.Clip) -> bool:
    if not clip.prompt_txt.is_file():
        return False
    head = clip.prompt_txt.read_bytes()[:64].decode("utf-8", errors="replace")
    return bool(EXPORTED_RE.match(head))


def _run_audit(args: argparse.Namespace) -> int:
    buckets: dict[str, list[str]] = {
        "captioned": [],
        "missing": [],
        "failed": [],
        "warned": [],
        "exported": [],
    }
    scores: list[float] = []
    for clip in layout.discover(args.clips):
        if _is_exported(clip):
            buckets["exported"].append(clip.name)
        if not clip.prompt.is_file():
            buckets["missing"].append(clip.name)
            continue
        try:
            caption = Caption.read(clip.prompt)
        except Exception:  # noqa: BLE001 - an unreadable caption is a failed one
            buckets["failed"].append(clip.name)
            continue
        buckets["captioned"].append(clip.name)
        checks = caption.checks or {}
        if checks.get("fail"):
            buckets["failed"].append(clip.name)
        elif checks.get("warn"):
            buckets["warned"].append(clip.name)
        score = caption.provenance.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

    summary = {name: len(values) for name, values in buckets.items()}
    summary["mean_score"] = round(sum(scores) / len(scores), 3) if scores else None
    if args.list:
        for name in buckets[args.list]:
            _say(name)
    else:
        _say(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, **buckets}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def _run_show(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from . import conditioning

    caption = Caption.read(args.prompt)
    compiled = caption.compiled or render.compile_all(caption)
    if args.conditioning:
        _say(conditioning.full_text(compiled.get("conditioning") or {}))
        _say("")

    if args.style == "cwm":
        caption = replace(caption, compiled=compiled)
        # The system prompt goes to stderr: it is a label for whoever wires up
        # FastVideo, and it must never end up concatenated onto the user
        # sentence by someone piping stdout into a file.
        print(f"system={cwm_export.system_id(caption)}", file=sys.stderr)
        _say(cwm_export.user_text(caption, variant="window", prose=args.prose))
        return 0

    block = compiled[args.style]
    _say(block["global"])
    if args.style == "timed":
        if block.get("facing"):
            _say(block["facing"])
        _say("")
        _say(block["script"])
    else:
        _say("")
        for item, text in zip(caption.grid.bins, block["bins"]):
            _say(f"[{item.start:g}s-{item.stop:g}s] {text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "captions": _run_captions,
        "captions-evidence": _run_evidence,
        "captions-recompile": _run_recompile,
        "captions-slice": _run_slice,
        "captions-export": _run_export,
        "captions-audit": _run_audit,
        "captions-show": _run_show,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
