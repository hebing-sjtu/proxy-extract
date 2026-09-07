"""A self-contained multimodal chat client: Vertex Gemini, or any OpenAI-compatible gateway.

This is a reduced, standalone rewrite of the `mllm` package that the sibling
`low_high_pipeline` runs in production. It lives here rather than being imported
from there because the compute nodes that caption this corpus cannot reach the
internal Git host, and a captioning run that depends on a repo you cannot clone
is a captioning run that does not happen.

What was deliberately left behind:

- **The DashScope backend.** It requires a vendored SDK, refuses to run outside
  a Kubernetes pod, and reaches into a helper module of the sibling repo to
  check. Every one of those is the coupling this move exists to remove.
- **The `shrink_media` retry hook.** It re-sends a smaller video when the
  request is too big, but nothing here ever supplied a shrink function, so it
  was plumbing with no implementation behind it. A 124-frame 1344x768 clip
  inlines to a few megabytes against a 15 MB ceiling; if that ever stops being
  true, `PayloadTooLarge` says so by name.

What was fixed on the way in: system messages now become Vertex's
`systemInstruction` instead of a leading `"model"` turn. See `parts.render_vertex`.

Everything here is standard library. Signing the service-account JWT shells out
to `openssl` so that captioning a corpus does not require a crypto wheel.
"""

from __future__ import annotations

from .client import (
    ChatRequest,
    ChatResponse,
    OpenAICompatClient,
    PayloadTooLarge,
    VertexClient,
    build_client,
    parse_json_object,
)
from .env import env_any, load_dotenv
from .http import HTTPStatusError, is_retryable
from .parts import Image, Message, Text, Video
from .vertex import AccessToken, VertexAuthError, load_service_account

__all__ = [
    "AccessToken",
    "ChatRequest",
    "ChatResponse",
    "HTTPStatusError",
    "Image",
    "Message",
    "OpenAICompatClient",
    "PayloadTooLarge",
    "Text",
    "VertexAuthError",
    "VertexClient",
    "Video",
    "build_client",
    "env_any",
    "is_retryable",
    "load_dotenv",
    "load_service_account",
    "parse_json_object",
]
