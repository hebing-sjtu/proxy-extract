"""Tests for the vendored transport.

This code used to live in another repo and was covered by that repo's tests,
which do not travel with it. What is checked here is what would otherwise only
be discovered against a live endpoint several minutes into a corpus run: the
shape of the request body, which failures are worth retrying, and whether the
service-account assertion is actually a valid signature.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from clip_prompts.llm import client as llm_client
from clip_prompts.llm import env as llm_env
from clip_prompts.llm import http, parts, vertex

# --- rendering ------------------------------------------------------------


def test_a_system_message_becomes_system_instruction_not_a_model_turn():
    """Gemini has no system turn, and the wrong fix here fails silently.

    Folding the system prompt into `contents` as a `"model"` entry reads to the
    model as something it already said, so the instructions come back
    paraphrased rather than obeyed - and the request still returns 200.
    """
    contents, system = parts.render_vertex(
        (parts.Message.system(parts.Text("SYS")), parts.Message.user(parts.Text("hi")))
    )
    assert system == {"parts": [{"text": "SYS"}]}
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert all(entry["role"] != "system" for entry in contents)


def test_contents_begin_with_the_user_even_when_a_system_prompt_leads():
    contents, _ = parts.render_vertex(
        (
            parts.Message.system(parts.Text("SYS")),
            parts.Message.user(parts.Text("ask")),
            parts.Message("assistant", (parts.Text("reply"),)),
            parts.Message.user(parts.Text("repair")),
        )
    )
    assert [entry["role"] for entry in contents] == ["user", "model", "user"]


def test_no_system_prompt_means_no_system_instruction_key():
    contents, system = parts.render_vertex((parts.Message.user(parts.Text("hi")),))
    assert system is None
    assert len(contents) == 1


def test_a_video_carries_its_sampling_rate_next_to_the_bytes(tmp_path):
    video = tmp_path / "rgb.mp4"
    video.write_bytes(b"\x00\x01\x02")
    contents, _ = parts.render_vertex(
        (parts.Message.user(parts.Video(video, fps=4.0)),)
    )
    part = contents[0]["parts"][0]
    assert part["video_metadata"] == {"fps": 4.0}
    assert part["inline_data"]["mime_type"] == "video/mp4"
    assert base64.b64decode(part["inline_data"]["data"]) == b"\x00\x01\x02"


def test_the_openai_rendering_inlines_the_same_bytes_as_a_data_uri(tmp_path):
    image = tmp_path / "sheet.png"
    image.write_bytes(b"\x89PNG")
    rendered = parts.render_openai((parts.Message.user(parts.Image(image)),))
    url = rendered[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG"


# --- retry policy ---------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_the_statuses_that_mean_try_again_are_retried(status):
    assert http.is_retryable(http.HTTPStatusError(status, "busy"))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_the_statuses_that_mean_stop_are_not(status):
    assert not http.is_retryable(http.HTTPStatusError(status, "nope"))


def test_a_dropped_connection_is_retryable():
    assert http.is_retryable(urllib.error.URLError("connection reset"))
    assert http.is_retryable(TimeoutError())


@pytest.mark.parametrize(
    "base",
    ["https://host", "https://host/", "https://host/v1", "https://host/v1/chat/completions"],
)
def test_every_way_someone_writes_a_gateway_url_lands_in_one_place(base):
    assert http.api_url(base, "chat/completions") == "https://host/v1/chat/completions"


# --- request bodies -------------------------------------------------------


def _vertex(**kwargs) -> llm_client.VertexClient:
    return llm_client.VertexClient(
        project="p", location="global", tokens=vertex.AccessToken(sa={}), **kwargs
    )


def _body(request) -> dict:
    return json.loads(_vertex()._body(request))


def test_gemini_3_8_gets_a_thinking_level_and_no_temperature():
    """3.8 Flash rejects an explicit temperature, so sending one 400s the run."""
    body = _body(
        llm_client.ChatRequest(
            model="gemini-3.8-flash",
            messages=(parts.Message.user(parts.Text("hi")),),
            temperature=0.2,
            max_tokens=8000,
            json_object=True,
        )
    )
    gen = body["generationConfig"]
    assert "temperature" not in gen
    assert gen["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert gen["responseMimeType"] == "application/json"
    assert gen["maxOutputTokens"] == 8000


def test_an_older_model_keeps_the_temperature_it_was_given():
    body = _body(
        llm_client.ChatRequest(
            model="gemini-2.5-pro",
            messages=(parts.Message.user(parts.Text("hi")),),
            temperature=0.2,
        )
    )
    assert body["generationConfig"]["temperature"] == 0.2
    assert "thinkingConfig" not in body["generationConfig"]


def test_extra_generation_config_is_merged_rather_than_replaced():
    body = _body(
        llm_client.ChatRequest(
            model="gemini-2.5-pro",
            messages=(parts.Message.user(parts.Text("hi")),),
            max_tokens=10,
            extra={"generationConfig": {"topK": 40}},
        )
    )
    assert body["generationConfig"] == {"topK": 40, "maxOutputTokens": 10}


def test_an_oversized_clip_is_named_as_such_before_it_is_sent(tmp_path):
    video = tmp_path / "rgb.mp4"
    video.write_bytes(b"\x00" * 4096)
    request = llm_client.ChatRequest(
        model="gemini-3.8-flash", messages=(parts.Message.user(parts.Video(video)),)
    )
    with pytest.raises(llm_client.PayloadTooLarge) as caught:
        _vertex(inline_limit_bytes=1024)._body(request)
    assert "MB" in str(caught.value)


# --- replies --------------------------------------------------------------


def test_a_reply_wrapped_in_prose_is_still_parsed():
    assert llm_client.parse_json_object('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_a_reply_with_no_object_in_it_is_an_error():
    with pytest.raises(RuntimeError, match="not JSON"):
        llm_client.parse_json_object("I cannot help with that.")


def test_vertex_text_is_joined_across_parts():
    data = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
    assert llm_client.extract_vertex_text(data) == "a\nb"


def test_a_blocked_reply_reads_as_empty_rather_than_crashing():
    assert llm_client.extract_vertex_text({"candidates": [{"finishReason": "SAFETY"}]}) == ""
    assert llm_client.extract_vertex_text({}) == ""


# --- credentials ----------------------------------------------------------


def test_a_pem_survives_the_round_trip_through_a_dotenv_file():
    """`.env` holds the key on one line, so the newlines arrive escaped."""
    escaped = "-----BEGIN PRIVATE KEY-----\\nabc\\ndef\\n-----END PRIVATE KEY-----"
    assert vertex.unescape_pem(f'"{escaped}",') == (
        "-----BEGIN PRIVATE KEY-----\nabc\ndef\n-----END PRIVATE KEY-----"
    )


def test_dotenv_never_overrides_something_already_exported(tmp_path, monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT_ID", "from-the-shell")
    (tmp_path / ".env").write_text("VERTEX_PROJECT_ID=from-the-file\nVERTEX_LOCATION=us-central1\n")
    llm_env.load_dotenv(tmp_path / ".env")
    assert os.environ["VERTEX_PROJECT_ID"] == "from-the-shell"
    assert os.environ["VERTEX_LOCATION"] == "us-central1"


def test_load_env_reports_which_files_it_read(tmp_path):
    (tmp_path / ".env").write_text("A=1\n")
    assert llm_env.load_env(tmp_path, tmp_path / "absent") == [tmp_path / ".env"]


VERTEX_VARS = (
    "VERTEX_SA_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "VERTEX_PROJECT_ID",
    "VERTEX_PROJECT",
    "VERTEX_CLIENT_EMAIL",
    "VERTEX_PRIVATE_KEY",
    "VERTEXT_KEY",
)


@pytest.fixture
def no_vertex_env(monkeypatch):
    for name in VERTEX_VARS:
        monkeypatch.delenv(name, raising=False)


def test_an_incomplete_service_account_names_the_variable_to_set(no_vertex_env):
    with pytest.raises(vertex.VertexAuthError, match="VERTEX_CLIENT_EMAIL"):
        vertex.load_service_account()


def test_the_error_names_the_alias_spellings_too(no_vertex_env):
    """The sibling repo's template says VERTEX_PROJECT, so people have that one."""
    with pytest.raises(vertex.VertexAuthError) as caught:
        vertex.load_service_account()
    assert "VERTEX_PROJECT_ID (or VERTEX_PROJECT)" in str(caught.value)
    assert "VERTEXT_KEY" in str(caught.value)
    assert "VERTEX_SA_JSON" in str(caught.value)


def test_a_half_filled_environment_says_which_half_arrived(no_vertex_env, monkeypatch):
    """Moving credentials between machines one variable at a time is the common case."""
    monkeypatch.setenv("VERTEX_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----x")
    with pytest.raises(vertex.VertexAuthError) as caught:
        vertex.load_service_account()
    message = str(caught.value)
    assert "already set: VERTEX_PRIVATE_KEY" in message
    assert "VERTEX_CLIENT_EMAIL" in message
    # The one that arrived must not also be listed as missing.
    assert "missing: VERTEX_PROJECT_ID (or VERTEX_PROJECT), VERTEX_CLIENT_EMAIL\n" in message


def test_a_key_file_supplies_all_three_at_once(no_vertex_env, monkeypatch, tmp_path):
    key_file = tmp_path / "sa.json"
    key_file.write_text(
        json.dumps(
            {
                "project_id": "proj",
                "client_email": "robot@proj.iam.gserviceaccount.com",
                "private_key": "-----BEGIN PRIVATE KEY-----\\nabc\\n-----END PRIVATE KEY-----",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_file))
    sa = vertex.load_service_account()
    assert sa["project_id"] == "proj"
    assert sa["token_uri"] == vertex.DEFAULT_TOKEN_URI
    assert "\n" in sa["private_key"]  # unescaped on the way out


def test_the_regional_and_global_endpoints_differ_by_host():
    assert vertex.generate_content_url("p", "global", "gemini-3.8-flash").startswith(
        "https://aiplatform.googleapis.com/"
    )
    assert vertex.generate_content_url("p", "us-central1", "gemini-3.8-flash").startswith(
        "https://us-central1-aiplatform.googleapis.com/"
    )


def test_a_fully_qualified_model_name_is_not_repeated_in_the_path():
    url = vertex.generate_content_url("p", "global", "publishers/google/models/gemini-3.8-flash")
    assert url.count("publishers/google/models") == 1


def test_a_cached_token_is_reused_and_a_stale_one_is_not(monkeypatch):
    """A whole-corpus run outlives the one-hour token, so this has to renew."""
    calls = []

    def fake_exchange(sa):
        calls.append(sa)
        return f"token-{len(calls)}", 1e12

    monkeypatch.setattr(vertex, "exchange_access_token", fake_exchange)
    token = vertex.AccessToken(sa={"client_email": "x"})
    assert token.get() == "token-1"
    assert token.get() == "token-1"
    token.invalidate()
    assert token.get() == "token-2"


def test_a_token_within_two_minutes_of_expiry_is_replaced_early(monkeypatch):
    monkeypatch.setattr(vertex, "exchange_access_token", lambda sa: ("fresh", 1e12))
    token = vertex.AccessToken(sa={}, token="stale", expires_at=0.0)
    assert token.get() == "fresh"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is what does the signing")
def test_the_assertion_we_send_is_a_signature_that_verifies(tmp_path):
    """Sign a real JWT and have openssl check it, because a bad signature is
    indistinguishable from bad credentials once Google has rejected it."""
    key = tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(key)],
        check=True,
        capture_output=True,
    )
    pub = tmp_path / "pub.pem"
    subprocess.run(
        ["openssl", "rsa", "-in", str(key), "-pubout", "-out", str(pub)],
        check=True,
        capture_output=True,
    )

    token = vertex.sign_jwt_rs256({"iss": "a@b.iam", "scope": "s"}, key.read_text())
    header, payload, signature = token.split(".")

    def unpad(chunk: str) -> bytes:
        return base64.urlsafe_b64decode(chunk + "=" * (-len(chunk) % 4))

    assert json.loads(unpad(header)) == {"alg": "RS256", "typ": "JWT"}
    assert json.loads(unpad(payload))["iss"] == "a@b.iam"

    (tmp_path / "sig").write_bytes(unpad(signature))
    verified = subprocess.run(
        [
            "openssl", "dgst", "-sha256", "-verify", str(pub),
            "-signature", str(tmp_path / "sig"),
        ],
        input=f"{header}.{payload}".encode("ascii"),
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr


def test_signing_leaves_no_key_behind_on_disk(tmp_path, monkeypatch):
    """The key is written out for openssl to read; it must not survive the call."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(vertex.VertexAuthError):
        vertex.sign_jwt_rs256({}, "-----BEGIN PRIVATE KEY-----\nnot-a-key\n-----END PRIVATE KEY-----")
    assert list(tmp_path.iterdir()) == []


# --- wiring ---------------------------------------------------------------


def test_a_gateway_without_a_url_says_which_variable_is_missing():
    with pytest.raises(RuntimeError, match="LITELLM_BASE_URL"):
        llm_client.build_client("litellm", api_key="k")


def test_an_unknown_backend_lists_the_ones_that_exist():
    with pytest.raises(RuntimeError, match=r"vertex\|litellm"):
        llm_client.build_client("dashscope")


def test_the_gateway_client_is_built_from_a_base_url():
    built = llm_client.build_client("litellm", api_key="k", base_url="https://host/v1")
    assert isinstance(built, llm_client.OpenAICompatClient)
    assert built.name == "litellm"


def test_the_package_imports_nothing_but_the_standard_library():
    """The whole point of the move.

    A dependency added here is one more thing to install on a node that cannot
    reach the internal index, so this asserts the property rather than trusting
    the next person to remember it. Docstrings may still name where the code
    came from; only imports are checked.
    """
    root = Path(vertex.__file__).parent
    outside: dict[str, str] = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # a relative import, i.e. our own siblings
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name and name not in sys.stdlib_module_names:
                    outside[name] = path.name
    assert outside == {}
