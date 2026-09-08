"""What a caption run tells you when it cannot find credentials.

This is the first thing that happens on a new node and the last thing anyone
wants to debug by reading source, so the diagnostics are worth pinning down.
"""

from __future__ import annotations

import pytest
from clip_prompts import observe
from clip_prompts.llm import VertexAuthError

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
def no_vertex_env(monkeypatch, tmp_path):
    """An environment with no credentials anywhere the search would reach.

    `observe.REPO` is repointed because a developer who has configured real
    credentials has a `.env.local` at the repository root, and without this the
    suite would pass or fail depending on whose machine it runs on.
    """
    for name in VERTEX_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(observe, "REPO", tmp_path / "repo")


def test_the_failure_says_where_it_looked_for_a_dotenv(no_vertex_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(VertexAuthError) as caught:
        observe.build_client("vertex")
    message = str(caught.value)
    assert "no .env file was found" in message
    assert str(tmp_path) in message
    assert ".env.example" in message


def test_a_dotenv_that_was_read_is_named_rather_than_the_search_path(
    no_vertex_env, tmp_path, monkeypatch
):
    """Found-but-incomplete and never-found are different problems."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text("VERTEX_PRIVATE_KEY=x\n")
    with pytest.raises(VertexAuthError) as caught:
        observe.build_client("vertex")
    message = str(caught.value)
    assert ".env files read:" in message
    assert str(tmp_path / ".env.local") in message
    assert "no .env file was found" not in message
    assert "already set: VERTEX_PRIVATE_KEY" in message


def test_an_extra_env_dir_is_searched_first(no_vertex_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / ".env").write_text("VERTEX_PROJECT=from-secrets\n")
    read = observe.load_env(secrets)
    assert read == [secrets / ".env"]
    assert observe.env_roots(secrets)[0] == secrets


def test_a_gateway_needs_no_service_account(no_vertex_env, tmp_path, monkeypatch):
    """The LiteLLM path is the fallback when a node has no GCP credentials."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    client = observe.build_client("litellm")
    assert client.base_url == "https://gateway.example/v1"
