"""One captioning round trip, plus the repair loop that makes it land.

The transport - Vertex or a LiteLLM gateway, service-account refresh, backoff
on 429, base64 inlining of video with a sampling rate - already exists in
`low_high_pipeline/src/mllm` and is in production on this corpus's sibling
pipeline. This module borrows it rather than growing a second copy, because
two retry policies against the same quota is a way to discover you had two.

What this module owns is the loop around it. The reply is parsed into a
`Caption`, checked against the grid, and if anything is structurally wrong the
model is told all of it and asked again with the whole conversation in context.
Two repairs by default: the first fixes real slips, the second is rare, and a
model still failing after that is failing at the task rather than at the
formatting, so the clip is better recorded as failed than coerced.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import prompts, vocab
from .contract import COMPILER_VERSION, Caption, Entity, Event, Scene
from .timeline import Grid

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_BACKEND = "vertex"

# The video is inlined as base64, so the sampling rate is a cost as much as a
# fidelity choice. Four frames a second resolves a footfall, which is the
# fastest thing a one-second grid needs to distinguish.
DEFAULT_SAMPLE_FPS = 4.0
DEFAULT_MAX_FRAMES = 48

DEFAULT_REPAIRS = 2

REPO = Path(__file__).resolve().parents[3]


def mllm_root(explicit: str | Path | None = None) -> Path:
    """Locate the `mllm` package, preferring an explicit override.

    Reported as a plain missing-directory error rather than an ImportError
    three frames deep, because the usual cause is that the sibling repo was
    not checked out and the fix is a path, not a pip install.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("CLIP_PROMPTS_MLLM", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(REPO / "low_high_pipeline" / "src")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "mllm").is_dir():
            return resolved
    raise FileNotFoundError(
        "cannot find the mllm package. Check out low_high_pipeline beside this "
        "repo, or point CLIP_PROMPTS_MLLM at the directory that contains it. "
        f"Looked in: {', '.join(str(c) for c in candidates)}"
    )


def load_env(explicit: str | Path | None = None) -> None:
    """Read the same .env files the sibling pipeline reads, then this repo's."""
    root = mllm_root(explicit)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mllm.config import load_dotenv

    for name in (".env", ".env.local"):
        for base in (root.parent, REPO, Path.cwd()):
            path = base / name
            if path.is_file():
                load_dotenv(path)


def build_client(backend: str = DEFAULT_BACKEND, *, mllm_src: str | Path | None = None):
    load_env(mllm_src)
    from mllm.client import build_client as _build

    if backend in {"litellm", "openai", "openai_compat", "openai-compat"}:
        return _build(
            backend,
            api_key=os.environ.get("LITELLM_API_KEY", ""),
            base_url=os.environ.get("LITELLM_BASE_URL", ""),
        )
    return _build(backend, api_key=os.environ.get("DASHSCOPE_API_KEY", ""))


@dataclass(frozen=True)
class Observation:
    caption: Caption
    attempts: int
    usage: dict
    quality: dict


def _parts(text: str, video: Path, sheet: Path | None, sample_fps: float, max_frames: int):
    from mllm.content import Image, Text, Video

    parts = [Text(text), Video(video, fps=sample_fps, max_frames=max_frames)]
    if sheet is not None:
        parts.append(Text("Contact sheet: one frame per bin, stamped with its bin index."))
        parts.append(Image(sheet))
    return parts


def to_caption(reply: dict, grid: Grid) -> Caption:
    """Turn a model reply into a `Caption`, without judging whether it is right.

    Camera events arrive in their own list because a model keeps two short
    lists straighter than one list with a discriminator field; they are merged
    into the single event stream here, which is where every consumer wants
    them.
    """
    scene = Scene.from_dict(reply.get("scene") or {})
    entities = tuple(Entity.from_dict(item) for item in reply.get("entities") or ())

    events: list[Event] = []
    for index, item in enumerate(reply.get("events") or ()):
        payload = dict(item)
        payload.setdefault("id", f"e{index}")
        payload["channel"] = "subject"
        events.append(Event.from_dict(payload))
    for index, item in enumerate(reply.get("camera") or ()):
        payload = dict(item)
        payload.setdefault("id", f"c{index}")
        payload["channel"] = "camera"
        payload["entity"] = None
        payload["facing"] = "unknown"
        events.append(Event.from_dict(payload))

    return Caption(
        grid=grid,
        scene=scene,
        entities=entities,
        events=tuple(events),
    ).bind()


def unknown_verbs(caption: Caption) -> list[str]:
    return [
        f"event {e.id!r} uses verb {e.verb!r}, which is not in the {e.channel} verb list"
        for e in caption.events
        if not vocab.known_verb(e.verb, channel=e.channel)
    ]


def observe(
    client,
    *,
    grid: Grid,
    video: Path,
    briefing: str,
    sheet: Path | None = None,
    model: str = DEFAULT_MODEL,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    repairs: int = DEFAULT_REPAIRS,
) -> Observation:
    """Ask once, repair up to `repairs` times, return a structurally valid caption."""
    from mllm.client import ChatRequest, parse_json_object
    from mllm.content import Message, Text

    text = prompts.instruction(grid, briefing, sheet=sheet is not None)
    history = [
        Message(role="system", parts=(Text(prompts.SYSTEM),)),
        Message.user(*_parts(text, video, sheet, sample_fps, max_frames)),
    ]

    totals: dict = {}
    last: list[str] = []
    caption: Caption | None = None
    quality: dict = {}

    for attempt in range(1, repairs + 2):
        response = client.complete(
            ChatRequest(
                model=model,
                messages=tuple(history),
                temperature=0.2,
                max_tokens=8000,
                json_object=True,
            )
        )
        for key, value in (response.usage or {}).items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value

        reply = parse_json_object(response.text)
        quality = dict(reply.get("quality") or {})
        caption = to_caption(reply, grid)
        last = caption.problems() + unknown_verbs(caption)
        if not last:
            return Observation(caption=caption, attempts=attempt, usage=totals, quality=quality)
        if attempt == repairs + 1:
            break
        history.append(Message(role="assistant", parts=(Text(response.text),)))
        history.append(Message.user(Text(prompts.repair(last))))

    raise RuntimeError(
        f"caption still invalid after {repairs + 1} attempts: {'; '.join(last)}"
    )


def provenance(*, model: str, backend: str, attempts: int, usage: dict, quality: dict) -> dict:
    return {
        "source": f"vlm:target:{model}",
        "backend": backend,
        "model": model,
        "prompt": "clip_prompts.timeline.v4",
        "compiler": COMPILER_VERSION,
        "attempts": attempts,
        "usage": usage,
        "quality": quality,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
