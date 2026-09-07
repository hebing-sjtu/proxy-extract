"""One API round trip, and the retry policy around it. No knowledge of any task.

`VertexClient` calls Vertex `generateContent` with a service account, which is
the default path. `OpenAICompatClient` targets a LiteLLM gateway or anything
else speaking `/v1/chat/completions`, which is the fallback when a node has a
gateway but no GCP credentials.

Both retry on the same rule and both refuse to send a request over the inline
ceiling, because the two failures that actually happen on a long run are a 429
and a clip that turned out bigger than expected.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .http import HTTPStatusError, api_url, is_retryable, post_json
from .parts import Message, render_openai, render_vertex
from .vertex import AccessToken, generate_content_url, load_service_account

# Vertex rejects a request whose inlined payload passes roughly 20 MB. The
# margin is for the base64 expansion and the JSON escaping already counted in
# `len(body)`, plus the prompt itself.
DEFAULT_INLINE_LIMIT = 15 * 1024 * 1024

OPENAI_COMPAT_NAMES = {"litellm", "openai", "openai_compat", "openai-compat"}
VERTEX_NAMES = {"vertex", "vertexai", "gcp"}


class PayloadTooLarge(RuntimeError):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(
            f"the request body is {size / 1e6:.1f} MB, over the {limit / 1e6:.1f} MB "
            "inline limit. The video is sent as base64, so this is the clip's own "
            "size: re-encode it smaller, or send the proxy instead of the target."
        )
        self.size = size
        self.limit = limit


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    max_tokens: int | None = None
    temperature: float | None = None
    json_object: bool = False
    # An escape hatch for per-model settings that do not deserve a field here.
    # Merged into the request body; a `generationConfig` key is merged into the
    # generation config rather than replacing it.
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: dict
    raw: dict


class Client(Protocol):
    name: str

    def complete(self, request: ChatRequest) -> ChatResponse: ...


def parse_json_object(text: str) -> dict:
    """Parse a JSON object out of a model reply, tolerating prose around it."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"the reply is not JSON: {text[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        # A RuntimeError rather than a TypeError: this is the model having
        # answered with the wrong shape, not the caller having passed one.
        raise RuntimeError(f"the reply is JSON but not an object: {text[:200]}")  # noqa: TRY004
    return parsed


def extract_openai_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and (part.get("type") == "text" or "text" in part):
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts).strip()
    return ""


def extract_vertex_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "\n".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()


def _sleep_for(attempt: int) -> None:
    time.sleep(5 * attempt)


@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str
    attempts: int = 3
    timeout_sec: float = 900.0
    inline_limit_bytes: int = DEFAULT_INLINE_LIMIT
    name: str = "litellm"

    def complete(self, request: ChatRequest) -> ChatResponse:
        body = self._body(request)
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                data = post_json(
                    api_url(self.base_url, "chat/completions"),
                    body,
                    api_key=self.api_key,
                    timeout_sec=self.timeout_sec,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.attempts and is_retryable(exc):
                    _sleep_for(attempt)
                    continue
                raise
            text = extract_openai_text(data)
            if text:
                usage = data.get("usage")
                return ChatResponse(
                    text=text, usage=usage if isinstance(usage, dict) else {}, raw=data
                )
            last_error = RuntimeError(f"the model returned no text: {data}")
            if attempt < self.attempts:
                _sleep_for(attempt)
        raise last_error  # type: ignore[misc]

    def _body(self, request: ChatRequest) -> bytes:
        payload: dict = {
            **request.extra,
            "model": request.model,
            "messages": render_openai(request.messages),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.json_object:
            payload["response_format"] = {"type": "json_object"}
        return _encode(payload, self.inline_limit_bytes)


@dataclass
class VertexClient:
    project: str
    location: str
    tokens: AccessToken
    attempts: int = 3
    timeout_sec: float = 900.0
    inline_limit_bytes: int = DEFAULT_INLINE_LIMIT
    name: str = "vertex"

    @classmethod
    def from_env(cls, *, attempts: int = 3, timeout_sec: float = 900.0) -> VertexClient:
        from .env import env_any

        sa = load_service_account()
        return cls(
            project=str(sa["project_id"]),
            location=env_any("VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION") or "global",
            tokens=AccessToken(sa=sa),
            attempts=attempts,
            timeout_sec=timeout_sec,
        )

    def complete(self, request: ChatRequest) -> ChatResponse:
        body = self._body(request)
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                data = post_json(
                    generate_content_url(self.project, self.location, request.model),
                    body,
                    api_key=self.tokens.get(),
                    timeout_sec=self.timeout_sec,
                )
            except HTTPStatusError as exc:
                last_error = exc
                # A 401 mid-run is the token having expired early, not the
                # credentials being wrong; drop it and the next attempt mints a
                # fresh one. If they really are wrong, the retries run out.
                if exc.status == 401:
                    self.tokens.invalidate()
                if attempt < self.attempts and (exc.status == 401 or is_retryable(exc)):
                    _sleep_for(attempt)
                    continue
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.attempts and is_retryable(exc):
                    _sleep_for(attempt)
                    continue
                raise
            text = extract_vertex_text(data)
            if text:
                usage = data.get("usageMetadata")
                return ChatResponse(
                    text=text, usage=usage if isinstance(usage, dict) else {}, raw=data
                )
            last_error = RuntimeError(f"Vertex returned no text: {data}")
            if attempt < self.attempts:
                _sleep_for(attempt)
        raise last_error  # type: ignore[misc]

    def _body(self, request: ChatRequest) -> bytes:
        extra = dict(request.extra)
        extra_gen = extra.pop("generationConfig", None)
        contents, system = render_vertex(request.messages)
        payload: dict = {**extra, "contents": contents}
        if system is not None:
            payload["systemInstruction"] = system

        gen: dict = dict(extra_gen) if isinstance(extra_gen, dict) else {}
        if request.max_tokens is not None:
            gen["maxOutputTokens"] = request.max_tokens
        # Gemini 3.8 Flash rejects an explicit temperature, and wants its
        # thinking budget named rather than left to default.
        if request.temperature is not None and not request.model.startswith("gemini-3.8"):
            gen["temperature"] = request.temperature
        if request.model.startswith("gemini-3.8"):
            gen["thinkingConfig"] = {"thinkingLevel": "LOW"}
        if request.json_object:
            gen["responseMimeType"] = "application/json"
        if gen:
            payload["generationConfig"] = gen
        return _encode(payload, self.inline_limit_bytes)


def _encode(payload: dict, limit: int) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    if len(body) > limit:
        raise PayloadTooLarge(len(body), limit)
    return body


def build_client(
    backend: str,
    *,
    api_key: str = "",
    base_url: str | None = None,
    attempts: int = 3,
    timeout_sec: float = 900.0,
) -> Client:
    key = backend.strip().lower()
    if key in VERTEX_NAMES:
        return VertexClient.from_env(attempts=attempts, timeout_sec=timeout_sec)
    if key in OPENAI_COMPAT_NAMES:
        if not base_url:
            raise RuntimeError(
                f"backend {backend!r} needs a base URL. Set LITELLM_BASE_URL, or "
                "use --backend vertex."
            )
        return OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            attempts=attempts,
            timeout_sec=timeout_sec,
            name="litellm" if key == "litellm" else "openai_compat",
        )
    raise RuntimeError(f"unknown backend {backend!r} (vertex|litellm)")
