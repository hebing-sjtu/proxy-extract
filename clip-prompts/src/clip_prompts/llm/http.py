"""Minimal stdlib HTTP, and the one opinion worth having about it: what to retry."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

# 429 and the 5xx family are the ones that come back on their own. 408/409/425
# are here because gateways in front of these APIs emit them under load and
# they mean the same thing: the request never reached the model.
RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:800]}")
        self.status = status
        self.body = body


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.status in RETRYABLE_HTTP
    return isinstance(exc, (TimeoutError, urllib.error.URLError))


def api_url(base_url: str, path: str) -> str:
    """Join a gateway base URL with `/v1/<path>`, tolerating either form.

    Accepts `https://host`, `https://host/v1`, or a full endpoint URL, because
    all three are what people actually paste into a config.
    """
    base = base_url.rstrip("/")
    path = path.strip("/")
    if base.endswith(f"/{path}"):
        return base
    if base.endswith("/v1"):
        return f"{base}/{path}"
    return f"{base}/v1/{path}"


def post_json(url: str, body: bytes, *, api_key: str, timeout_sec: float) -> dict:
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise HTTPStatusError(exc.code, exc.read().decode("utf-8", errors="ignore")) from exc
