"""One captioning round trip, plus the repair loop that makes it land.

The transport - Vertex or a LiteLLM gateway, service-account refresh, backoff
on 429, base64 inlining of video with a sampling rate - lives in `.llm`, which
is this repo's own copy of a client proven on the sibling pipeline. It is a
copy rather than an import because the nodes that run this cannot reach the
host that the original lives on.

What this module owns is the loop around it. The reply is parsed into a
`Caption`, checked against the grid, and if anything is structurally wrong the
model is told all of it and asked again with the whole conversation in context.
Two repairs by default: the first fixes real slips, the second is rare, and a
model still failing after that is failing at the task rather than at the
formatting, so the clip is better recorded as failed than coerced.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import prompts, vocab
from .contract import COMPILER_VERSION, Caption, Entity, Event, Scene
from .llm import ChatRequest, Image, Message, Text, Video, parse_json_object
from .llm import build_client as _build_client
from .llm.env import load_env as _load_env
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


def load_env(extra: str | Path | None = None) -> list[Path]:
    """Load credentials from `.env` files, and report which ones were read.

    The sibling checkout is searched last and only if it happens to be there.
    That is a convenience for whoever still has their keys in it, not a
    dependency: nothing breaks when it is absent, which on the caption nodes it
    always is.
    """
    roots = [REPO, Path.cwd(), REPO / "low_high_pipeline"]
    if extra:
        roots.insert(0, Path(extra).expanduser())
    return _load_env(*roots)


def build_client(backend: str = DEFAULT_BACKEND, *, env_dir: str | Path | None = None):
    load_env(env_dir)
    return _build_client(
        backend,
        api_key=os.environ.get("LITELLM_API_KEY", ""),
        base_url=os.environ.get("LITELLM_BASE_URL", ""),
    )


@dataclass(frozen=True)
class Observation:
    caption: Caption
    attempts: int
    usage: dict
    quality: dict


def _parts(text: str, video: Path, sheet: Path | None, sample_fps: float, max_frames: int):
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
