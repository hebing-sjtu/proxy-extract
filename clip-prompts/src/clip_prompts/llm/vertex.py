"""Service-account OAuth for Vertex AI, using `openssl` instead of a crypto wheel.

Google's own client libraries would do this, but they pull in a dependency tree
that a captioning job has no other use for, and on a locked-down node the
install is the part that fails. Signing one RS256 assertion is a few lines and
an `openssl` invocation.

The private key never reaches a log or an exception message.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .env import env_any

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
DEFAULT_LOCATION = "global"


class VertexAuthError(RuntimeError):
    pass


def unescape_pem(value: str) -> str:
    """Undo the manglings a PEM picks up on its way through a `.env` file."""
    text = value.strip().rstrip(",").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.replace("\\n", "\n").replace("\\t", "\t").strip()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_jwt_rs256(claims: dict, private_key_pem: str) -> str:
    if shutil.which("openssl") is None:
        raise VertexAuthError(
            "openssl is not on PATH, and it is what signs the Vertex service-account "
            "assertion. Install it, or use the LiteLLM backend, which authenticates "
            "with a bearer token and needs no signing."
        )
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps(claims, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode("ascii")
    key = unescape_pem(private_key_pem)
    if "BEGIN" not in key:
        raise VertexAuthError("the Vertex private key is not a PEM block")
    # Written to a file because openssl reads the key from a path, and mode 600
    # before anything is in it would be better still - but the handle is already
    # ours alone, and the file is unlinked in the `finally`.
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write(key if key.endswith("\n") else key + "\n")
        key_path = handle.name
    try:
        os.chmod(key_path, 0o600)
        proc = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_path],
            input=signing_input,
            capture_output=True,
            check=False,
        )
    finally:
        Path(key_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", errors="ignore")[-200:]
        raise VertexAuthError(f"openssl failed to sign the service-account JWT: {detail}")
    return f"{header}.{payload}.{_b64url(proc.stdout)}"


def load_service_account() -> dict:
    """Assemble a service-account dict from a JSON file or from split env vars.

    File, preferred: `VERTEX_SA_JSON` or `GOOGLE_APPLICATION_CREDENTIALS`.

    Split env, which is what most `.env` files here already hold::

        VERTEX_PROJECT_ID / VERTEX_PROJECT
        VERTEX_CLIENT_EMAIL
        VERTEX_PRIVATE_KEY / VERTEXT_KEY
        VERTEX_PRIVATE_KEY_ID / VERTEXT_KEY_ID

    The `VERTEXT_*` spellings are a typo that predates this code and is baked
    into credentials people already have. Reading both costs one tuple.
    """
    json_path = env_any("VERTEX_SA_JSON", "GOOGLE_APPLICATION_CREDENTIALS")
    if json_path:
        path = Path(json_path).expanduser()
        if not path.is_file():
            raise VertexAuthError(f"VERTEX_SA_JSON points at a file that is not there: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise VertexAuthError(f"{path} is not a JSON object")
    else:
        data = {
            "type": "service_account",
            "project_id": env_any("VERTEX_PROJECT_ID", "VERTEX_PROJECT"),
            "client_email": env_any("VERTEX_CLIENT_EMAIL"),
            "private_key": env_any("VERTEX_PRIVATE_KEY", "VERTEXT_KEY"),
            "private_key_id": env_any("VERTEX_PRIVATE_KEY_ID", "VERTEXT_KEY_ID"),
            "token_uri": env_any("VERTEX_TOKEN_URI", default=DEFAULT_TOKEN_URI),
        }
    required = (
        ("project_id", "VERTEX_PROJECT_ID", "VERTEX_PROJECT"),
        ("client_email", "VERTEX_CLIENT_EMAIL", None),
        ("private_key", "VERTEX_PRIVATE_KEY", "VERTEXT_KEY"),
    )
    missing = [names for names in required if not str(data.get(names[0]) or "").strip()]
    if missing:
        # Naming what *was* found matters as much as what was not: a half-filled
        # environment looks identical to an empty one from the error alone, and
        # the half-filled case is the common one when credentials are being
        # moved between machines a variable at a time.
        found = [names[1] for names in required if names not in missing]
        wanted = ", ".join(
            variable if alias is None else f"{variable} (or {alias})"
            for _, variable, alias in missing
        )
        raise VertexAuthError(
            "the Vertex service account is incomplete.\n"
            f"  missing: {wanted}\n"
            + (f"  already set: {', '.join(found)}\n" if found else "")
            + "  Simplest fix: put the key file somewhere readable and set "
            "VERTEX_SA_JSON (or GOOGLE_APPLICATION_CREDENTIALS) to its path, "
            "which supplies all three at once."
        )
    data["private_key"] = unescape_pem(str(data["private_key"]))
    data.setdefault("token_uri", DEFAULT_TOKEN_URI)
    return data


def exchange_access_token(sa: dict) -> tuple[str, float]:
    now = int(time.time())
    token_uri = str(sa.get("token_uri") or DEFAULT_TOKEN_URI)
    assertion = sign_jwt_rs256(
        {
            "iss": sa["client_email"],
            "sub": sa["client_email"],
            "aud": token_uri,
            "iat": now,
            "exp": now + 3600,
            "scope": CLOUD_PLATFORM_SCOPE,
        },
        str(sa["private_key"]),
    )
    body = urllib.parse.urlencode(
        {"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion}
    ).encode("ascii")
    request = urllib.request.Request(
        token_uri,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")[:400]
        raise VertexAuthError(f"token exchange HTTP {exc.code}: {raw}") from exc
    token = payload.get("access_token")
    if not token:
        raise VertexAuthError("token exchange returned no access_token")
    expires = now + int(payload.get("expires_in") or 3600)
    return str(token), float(expires)


@dataclass
class AccessToken:
    """A cached bearer token that refreshes itself two minutes before it expires.

    A whole-corpus run outlives the one-hour token, so this has to renew mid-run
    rather than at startup.
    """

    sa: dict
    token: str = ""
    expires_at: float = 0.0

    def get(self, *, force: bool = False) -> str:
        if force or not self.token or time.time() >= self.expires_at - 120:
            self.token, self.expires_at = exchange_access_token(self.sa)
        return self.token

    def invalidate(self) -> None:
        self.token = ""
        self.expires_at = 0.0


def generate_content_url(project: str, location: str, model: str) -> str:
    model = model.removeprefix("models/").removeprefix("publishers/google/models/")
    host = (
        "https://aiplatform.googleapis.com"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com"
    )
    return (
        f"{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )
