"""Whether a clip's depth and semantics are fit to train on, judged by a VLM.

The extraction can fail in ways no structural audit sees: a region whose depth
pulses from frame to frame, a road that is vegetation for six frames, a person
the segmenter never found. Each frame is valid on its own; the defect is in the
sequence, or in the relation to the picture. So every clip gets a review panel -
RGB, depth, semantics and the semantics over the RGB, side by side - and a VLM
watches it and scores the two tracks for stability and for correctness.

Two numbers per clip are also measured here, for sorting and for checking the
VLM against something it cannot talk its way around. Both use three-frame
windows so that smooth camera motion cancels and only flicker remains:

- `depth_jitter`: the median over pixels of |c[t+1] - 2 c[t] + c[t-1]| in log-depth
  codes (one code is 4.4% of a distance), averaged over frames.
- `semantic_flicker`: the share of pixels whose label at t differs from the label
  at t-1 and t+1 while those two agree - a one-frame flash.

What is written, per clip:

    annotations/quality_panel.mp4   the 2x2 panel the VLM saw
    annotations/quality.json        scores, issues, metrics, and the decision
    annotations/quality_override.json  a human's verdict, which wins when present
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .layout import Clip
from .llm import ChatRequest, Message, Text, Video, parse_json_object

PROMPT_VERSION = "clip_prompts.quality.v1"
QUALITY_NAME = "quality.json"
OVERRIDE_NAME = "quality_override.json"
PANEL_NAME = "quality_panel.mp4"
AUDIT_NAME = "quality_audit.json"

TILE_W, TILE_H = 640, 360
PANEL_FPS = 24.0
# Flicker lives between neighbouring frames, so the VLM needs a dense sample:
# at 8 fps a one-frame flash at 24 fps is caught one time in three, and a
# defect lasting a few frames almost always.
DEFAULT_SAMPLE_FPS = 8.0
DEFAULT_MAX_FRAMES = 48

SKY_CODE = 255
SCORE_KEYS = ("depth_temporal", "depth_accuracy", "semantic_temporal", "semantic_accuracy")
TRACKS = ("depth", "semantic")
KINDS = ("flicker", "scale_jump", "wrong_label", "missing_object", "boundary", "hole", "other")
SEVERITIES = ("minor", "major")
VERDICTS = ("accept", "reject")

CLASS_NAMES = (
    "void_unknown", "sky", "water", "terrain", "road_paved", "vegetation",
    "building_structure", "infrastructure", "human", "animal", "vehicle", "prop",
)
PALETTE = np.array(
    [
        (0, 0, 0), (110, 180, 240), (30, 90, 200), (150, 110, 70), (128, 128, 128),
        (40, 160, 50), (200, 90, 60), (230, 200, 40), (255, 40, 200), (255, 140, 0),
        (0, 230, 230), (170, 90, 230),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class Thresholds:
    """The rule that turns a VLM reply into accept or reject."""

    min_score: int = 3
    reject_major: bool = True
    trust_verdict: bool = True

    def as_dict(self) -> dict:
        return {
            "min_score": self.min_score,
            "reject_major": self.reject_major,
            "trust_verdict": self.trust_verdict,
        }


DEFAULT_THRESHOLDS = Thresholds()


# ------------------------------------------------------------------ the panel


def semantic_png(clip: Clip, index: int) -> Path:
    return clip.root / "duv" / f"{index:06d}.semantic_id.png"


def panel_path(clip: Clip) -> Path:
    return clip.annotations / PANEL_NAME


def colour_depth(codes: np.ndarray) -> np.ndarray:
    """Log-depth codes to RGB: warm is near, blue is far, black is sky or no depth."""
    import cv2

    near_high = (254 - np.minimum(codes, 254)).astype(np.uint8)
    rgb = cv2.applyColorMap(near_high, cv2.COLORMAP_TURBO)[:, :, ::-1]
    rgb = np.ascontiguousarray(rgb)
    rgb[codes >= SKY_CODE] = 0
    return rgb


def colour_semantic(ids: np.ndarray) -> np.ndarray:
    return PALETTE[np.minimum(ids, len(PALETTE) - 1)]


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    import cv2

    cv2.rectangle(tile, (0, 0), (12 + 11 * len(text), 24), (0, 0, 0), thickness=-1)
    cv2.putText(tile, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def compose(rgb: np.ndarray, codes: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """One 2x2 panel frame from tile-sized RGB, depth codes and class ids."""
    sem = colour_semantic(ids)
    overlay = (rgb.astype(np.uint16) + sem.astype(np.uint16)) // 2
    top = np.concatenate([_label(rgb.copy(), "RGB"), _label(colour_depth(codes), "DEPTH")], axis=1)
    bottom = np.concatenate(
        [_label(sem.copy(), "SEMANTIC"), _label(overlay.astype(np.uint8), "SEMANTIC OVER RGB")],
        axis=1,
    )
    return np.concatenate([top, bottom], axis=0)


def _video_frames(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            yield bgr[:, :, ::-1]
    finally:
        capture.release()


def _semantic_frames(clip: Clip, duv_frames):
    """cwm12 ids per frame from duv/, or from nothing if the clip has no per-frame form."""
    import cv2

    for index, duv in enumerate(duv_frames):
        png = semantic_png(clip, index)
        ids = cv2.imread(str(png), cv2.IMREAD_UNCHANGED) if png.is_file() else None
        if ids is None:
            raise FileNotFoundError(f"{png} is missing; the quality panel needs duv/ semantic ids")
        yield duv, ids


def _fit(array: np.ndarray, *, nearest: bool) -> np.ndarray:
    import cv2

    if array.shape[:2] == (TILE_H, TILE_W):
        return array
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(array, (TILE_W, TILE_H), interpolation=interpolation)


@dataclass
class _Temporal:
    """Three-frame flicker measures, streamed so a clip costs three frames of memory."""

    codes: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    jitter: list = field(default_factory=list)
    flashes: list = field(default_factory=list)
    medians: list = field(default_factory=list)

    def add(self, codes: np.ndarray, ids: np.ndarray) -> None:
        valid = codes < SKY_CODE
        self.medians.append(float(np.median(codes[valid])) if valid.any() else float("nan"))
        self.codes = [*self.codes[-2:], codes.astype(np.int16)]
        self.ids = [*self.ids[-2:], ids]
        if len(self.codes) < 3:
            return
        before, now, after = self.codes
        both = (before < SKY_CODE) & (now < SKY_CODE) & (after < SKY_CODE)
        if both.any():
            self.jitter.append(float(np.median(np.abs(after - 2 * now + before)[both])))
        a, b, c = self.ids
        self.flashes.append(float(((a == c) & (b != a)).mean()))

    def result(self) -> dict:
        medians = np.asarray(self.medians, dtype=np.float64)
        steps = np.abs(np.diff(medians[~np.isnan(medians)]))
        return {
            "frames": len(self.medians),
            "depth_jitter": round(float(np.mean(self.jitter)), 4) if self.jitter else None,
            "depth_jitter_max": round(float(np.max(self.jitter)), 4) if self.jitter else None,
            "depth_median_step_max": round(float(steps.max()), 3) if steps.size else None,
            "semantic_flicker": round(float(np.mean(self.flashes)), 6) if self.flashes else None,
            "semantic_flicker_max": round(float(np.max(self.flashes)), 6) if self.flashes else None,
        }


def ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - absent or broken, fall back to PATH
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("no ffmpeg: pip install imageio-ffmpeg") from None
        return found


def render_panel(clip: Clip, out: Path | None = None) -> tuple[Path, dict]:
    """Write the review panel and return it with the clip's flicker measures."""
    out = Path(out) if out else panel_path(clip)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.mp4")
    command = [
        ffmpeg_binary(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * TILE_W}x{2 * TILE_H}",
        "-r", f"{PANEL_FPS:g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
    ]
    temporal = _Temporal()
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        pairs = zip(_video_frames(clip.rgb), _semantic_frames(clip, _video_frames(clip.duv)))
        for rgb, (duv, ids) in pairs:
            rgb_t = _fit(np.ascontiguousarray(rgb), nearest=False)
            codes_t = _fit(np.ascontiguousarray(duv[:, :, 0]), nearest=True)
            ids_t = _fit(ids, nearest=True)
            temporal.add(codes_t, ids_t)
            encoder.stdin.write(compose(rgb_t, codes_t, ids_t).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {encoder.stderr.read().decode(errors='replace')[-400:]}")
    except BaseException:
        encoder.kill()
        partial.unlink(missing_ok=True)
        raise
    metrics = temporal.result()
    if not metrics["frames"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{clip.name}: no frames decoded from {clip.rgb} and {clip.duv}")
    partial.replace(out)
    return out, metrics


# ------------------------------------------------------------------ the judge

SYSTEM = f"""You review automatically extracted depth and semantic segmentation for
game-video clips that will be used to train a video model. Defects you let through
are learned by the model, so judge strictly but fairly.

The video is a 2x2 panel, all four tiles showing the same frames:
- top-left RGB: the source video.
- top-right DEPTH: warm (red/yellow) is near, blue is far, black is sky or no depth.
- bottom-left SEMANTIC: one flat colour per class. Classes and colours:
  {", ".join(f"{name}=rgb{tuple(int(v) for v in PALETTE[i])}" for i, name in enumerate(CLASS_NAMES))}.
- bottom-right: the semantic colours blended 50/50 over the RGB, for checking alignment.

Score each of the four aspects from 1 to 5:
- depth_temporal: depth is stable over time. Flicker is a change between neighbouring
  frames that camera or object motion does not explain: a region pulsing, the whole map
  brightening and darkening (scale jumps), patches appearing for a few frames.
- depth_accuracy: depth matches the picture. Near things warm and far things blue, the
  ordering of objects right, sky black and nothing else black, edges on the RGB edges,
  no holes.
- semantic_temporal: labels are stable over time. No region or object flashing between
  classes, no labels appearing or vanishing for a few frames.
- semantic_accuracy: labels are right. Large regions in the right class (road is not
  vegetation, sky is not building), people and vehicles found, boundaries on the RGB
  boundaries.

Scale: 5 no visible problem; 4 small problem that would not hurt training; 3 noticeable
but local or brief; 2 obvious over a large region or a long time; 1 unusable.
On-screen HUD or UI elements may carry any label; do not count them.

Reply with only a JSON object:
{{"depth_temporal": int, "depth_accuracy": int, "semantic_temporal": int,
 "semantic_accuracy": int,
 "issues": [{{"track": "depth"|"semantic", "kind": {"|".join(f'"{k}"' for k in KINDS)},
             "severity": "minor"|"major", "start": seconds, "end": seconds, "note": str}}],
 "verdict": "accept"|"reject",
 "summary": "one sentence"}}
"major" means the defect should keep the clip out of training. "verdict" is reject if
you would keep the clip out of training."""


class ReplyError(ValueError):
    pass


def validate(reply: dict, duration: float) -> dict:
    """The reply in canonical form, or ReplyError naming everything wrong with it."""
    problems: list[str] = []
    scores: dict[str, int] = {}
    for key in SCORE_KEYS:
        value = reply.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            problems.append(f"{key} must be an integer from 1 to 5, got {value!r}")
        else:
            scores[key] = round(value)
    verdict = str(reply.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        problems.append(f"verdict must be accept or reject, got {reply.get('verdict')!r}")
    issues = []
    raw = reply.get("issues", [])
    if not isinstance(raw, list):
        problems.append("issues must be a list")
        raw = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"issues[{position}] must be an object")
            continue
        track = str(item.get("track", "")).lower()
        severity = str(item.get("severity", "")).lower()
        if track not in TRACKS or severity not in SEVERITIES:
            problems.append(f"issues[{position}] needs track in {TRACKS} and severity in {SEVERITIES}")
            continue
        kind = str(item.get("kind", "other")).lower()
        start, end = _seconds(item.get("start"), duration), _seconds(item.get("end"), duration)
        if start is not None and end is not None and end < start:
            start, end = end, start
        issues.append(
            {
                "track": track,
                "kind": kind if kind in KINDS else "other",
                "severity": severity,
                "start": start,
                "end": end,
                "note": str(item.get("note", "")).strip(),
            }
        )
    if problems:
        raise ReplyError("; ".join(problems))
    return {
        "scores": scores,
        "issues": issues,
        "verdict": verdict,
        "summary": str(reply.get("summary", "")).strip(),
    }


def _seconds(value, duration: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(min(max(float(value), 0.0), duration), 2)


def decide(vlm: dict, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> dict:
    """Accept or reject, with every reason that applied."""
    reasons = []
    if thresholds.trust_verdict and vlm["verdict"] == "reject":
        reasons.append("vlm verdict reject")
    low = [key for key, value in vlm["scores"].items() if value < thresholds.min_score]
    reasons.extend(f"{key} < {thresholds.min_score}" for key in low)
    if thresholds.reject_major:
        majors = sorted({f"{i['track']} {i['kind']}" for i in vlm["issues"] if i["severity"] == "major"})
        reasons.extend(f"major {name}" for name in majors)
    score = sum(vlm["scores"].values()) / (5 * len(SCORE_KEYS))
    return {
        "verdict": "reject" if reasons else "accept",
        "score": round(score, 3),
        "reasons": reasons,
        "thresholds": thresholds.as_dict(),
    }


def ask(client, model: str, panel: Path, duration: float, *, sample_fps: float, attempts: int = 2):
    """One judged reply, repaired once if its shape is wrong."""
    messages = [
        Message.system(Text(SYSTEM)),
        Message.user(
            Text(f"Review this {duration:.2f}-second clip."),
            Video(panel, fps=sample_fps, max_frames=DEFAULT_MAX_FRAMES),
        ),
    ]
    usage: dict = {}
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response = client.complete(
            ChatRequest(model=model, messages=tuple(messages), temperature=0.0, json_object=True)
        )
        for key, value in (response.usage or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        try:
            return validate(parse_json_object(response.text), duration), attempt, usage
        except (ReplyError, RuntimeError, ValueError) as exc:
            error = exc
            messages += [
                Message("assistant", (Text(response.text),)),
                Message.user(Text(f"That reply is invalid: {exc}. Reply again with only the JSON object.")),
            ]
    raise ReplyError(f"no valid reply after {attempts} attempts: {error}")


# ------------------------------------------------------------- per-clip files


def quality_path(clip: Clip) -> Path:
    return clip.annotations / QUALITY_NAME


def override_path(clip: Clip) -> Path:
    return clip.annotations / OVERRIDE_NAME


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def judge_clip(
    clip: Clip,
    client,
    *,
    model: str,
    backend: str,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    redo: bool = False,
) -> dict:
    """Panel, measures, VLM, decision; written to annotations/quality.json."""
    existing = read_json(quality_path(clip))
    if existing and existing.get("prompt") == PROMPT_VERSION and not redo:
        return {"clip": clip.name, "status": "reused", "verdict": existing["decision"]["verdict"]}
    clip.check()
    report = clip.report()
    duration = float(report.get("frames", 0)) / float(report.get("fps") or PANEL_FPS)
    panel, metrics = render_panel(clip)
    vlm, attempts, usage = ask(client, model, panel, duration, sample_fps=sample_fps)
    decision = decide(vlm, thresholds)
    write_json(
        quality_path(clip),
        {
            "clip": clip.name,
            "prompt": PROMPT_VERSION,
            "model": model,
            "backend": backend,
            "sample_fps": sample_fps,
            "duration": round(duration, 3),
            "attempts": attempts,
            "usage": usage,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "panel": str(panel.relative_to(clip.root)),
            "vlm": vlm,
            "metrics": metrics,
            "decision": decision,
        },
    )
    return {
        "clip": clip.name,
        "status": "written",
        "verdict": decision["verdict"],
        "score": decision["score"],
        "reasons": decision["reasons"],
    }


def set_override(clip: Clip, verdict: str | None, note: str = "") -> dict | None:
    """Record a human verdict, or clear it with None."""
    path = override_path(clip)
    if verdict is None:
        path.unlink(missing_ok=True)
        return None
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    payload = {
        "verdict": verdict,
        "note": note.strip(),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(path, payload)
    return payload


# ------------------------------------------------------------------- summary

CLIP_NAME = re.compile(r"^(clip|seg)_[A-Za-z0-9_]+$")


def summarize(root: Path) -> dict:
    """One clip's row for the audit and the workbench, from files alone."""
    clip = Clip(Path(root))
    quality = read_json(quality_path(clip))
    override = read_json(override_path(clip))
    row: dict = {
        "clip": clip.name,
        "complete": clip.report_path.is_file(),
        "judged": bool(quality),
        "override": override,
    }
    if quality:
        decision = quality.get("decision") or {}
        vlm = quality.get("vlm") or {}
        issues = vlm.get("issues") or []
        row.update(
            {
                "vlm_verdict": vlm.get("verdict"),
                "decision": decision.get("verdict"),
                "score": decision.get("score"),
                "reasons": decision.get("reasons") or [],
                "scores": vlm.get("scores") or {},
                "major": sum(1 for i in issues if i.get("severity") == "major"),
                "minor": sum(1 for i in issues if i.get("severity") == "minor"),
                "metrics": quality.get("metrics") or {},
                "model": quality.get("model"),
                "written": quality.get("written"),
                "prompt": quality.get("prompt"),
            }
        )
    final = (override or {}).get("verdict") or row.get("decision")
    row["final"] = final if row["complete"] else None
    return row


def scan(root: Path, *, workers: int = 32) -> list[dict]:
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and CLIP_NAME.match(p.name))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(summarize, dirs))


def audit(rows: list[dict]) -> dict:
    complete = [r for r in rows if r["complete"]]
    lists = {
        "accepted": [r["clip"] for r in complete if r["final"] == "accept"],
        "rejected": [r["clip"] for r in complete if r["final"] == "reject"],
        "unjudged": [r["clip"] for r in complete if not r["judged"] and not r["override"]],
        "overridden": [r["clip"] for r in complete if r["override"]],
    }
    reasons: dict[str, int] = {}
    for row in complete:
        for reason in row.get("reasons") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    scores = [r["score"] for r in complete if isinstance(r.get("score"), (int, float))]
    summary = {
        "clips": len(rows),
        "complete": len(complete),
        "judged": sum(1 for r in complete if r["judged"]),
        **{key: len(value) for key, value in lists.items()},
        "mean_score": round(float(np.mean(scores)), 3) if scores else None,
        "reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        "prompt": PROMPT_VERSION,
    }
    return {"summary": summary, **lists}
