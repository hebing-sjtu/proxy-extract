"""Backend-neutral message parts, and their per-wire-format renderings.

A caller says what it wants to send - `Text`, `Image`, `Video` - and the client
decides how that travels. Keeping both renderings side by side here is what
lets one prompt run against Vertex or against a gateway without the caller
knowing which.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Image:
    path: Path
    detail: str | None = None


@dataclass(frozen=True)
class Video:
    path: Path
    fps: float | None = None
    max_frames: int | None = None
    detail: str | None = None


Part = Text | Image | Video


@dataclass(frozen=True)
class Message:
    role: str
    parts: tuple[Part, ...]

    @classmethod
    def user(cls, *parts: Part) -> Message:
        return cls("user", tuple(parts))

    @classmethod
    def system(cls, *parts: Part) -> Message:
        return cls("system", tuple(parts))


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".mp4":
        return "video/mp4"
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def data_uri(path: Path) -> str:
    return f"data:{guess_mime(path)};base64,{_b64(path)}"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def render_openai(messages: tuple[Message, ...]) -> list[dict]:
    """OpenAI chat format, with every byte inlined as a base64 data URI."""
    rendered: list[dict] = []
    for message in messages:
        content: list[dict] = []
        for part in message.parts:
            if isinstance(part, Text):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, Image):
                image_url: dict = {"url": data_uri(part.path)}
                if part.detail:
                    image_url["detail"] = part.detail
                content.append({"type": "image_url", "image_url": image_url})
            else:
                file_part: dict = {
                    "file_data": data_uri(part.path),
                    "format": guess_mime(part.path),
                }
                if part.detail:
                    file_part["detail"] = part.detail
                if part.fps is not None:
                    file_part["video_metadata"] = {"fps": part.fps}
                content.append({"type": "file", "file": file_part})
        rendered.append({"role": message.role, "content": content})
    return rendered


def _vertex_parts(parts: tuple[Part, ...]) -> list[dict]:
    out: list[dict] = []
    for part in parts:
        if isinstance(part, Text):
            out.append({"text": part.text})
            continue
        inline = {"mime_type": guess_mime(part.path), "data": _b64(part.path)}
        if isinstance(part, Video) and part.fps is not None:
            out.append({"inline_data": inline, "video_metadata": {"fps": part.fps}})
        else:
            out.append({"inline_data": inline})
    return out


def render_vertex(messages: tuple[Message, ...]) -> tuple[list[dict], dict | None]:
    """Split messages into Vertex `contents` and its separate `systemInstruction`.

    Gemini does not have a system *turn*. It has a top-level `systemInstruction`
    field, and `contents` is a user/model alternation that has to begin with the
    user. Folding a system message into `contents` under either role is wrong in
    a way that does not raise: as `"model"` it reads as something Gemini already
    said, so the instructions come back paraphrased instead of obeyed.

    Returning the two pieces separately is what forces the caller to put each
    one where it belongs.
    """
    contents: list[dict] = []
    system: list[dict] = []
    for message in messages:
        if message.role == "system":
            system.extend(_vertex_parts(message.parts))
            continue
        role = "user" if message.role == "user" else "model"
        contents.append({"role": role, "parts": _vertex_parts(message.parts)})
    return contents, ({"parts": system} if system else None)
