"""The Docker / runbook packaging stays wired to the code it claims to wrap.

A mismatch here is the kind that only shows up on a GPU node: the image builds,
`fetch` downloads the wrong checkpoint, and eight workers then fail identically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_fetch_sets_use_the_same_repo_names_the_backends_do():
    from proxy_extract.depth.depth_anything import INDOOR_CHECKPOINT, OUTDOOR_CHECKPOINT
    from proxy_extract.depth.mapanything import APACHE_CHECKPOINT, DEFAULT_CHECKPOINT
    from proxy_extract.semantic.panoptic import ADE20K_CHECKPOINT, CITYSCAPES_CHECKPOINT

    # Imported the way the container does, so a rename in either place fails.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fetch_models", ROOT / "docker" / "fetch_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SETS["default"] == (ADE20K_CHECKPOINT, OUTDOOR_CHECKPOINT)
    assert ADE20K_CHECKPOINT in module.SETS["semantic"]
    assert CITYSCAPES_CHECKPOINT in module.SETS["semantic"]
    assert {INDOOR_CHECKPOINT, OUTDOOR_CHECKPOINT} <= set(module.SETS["depth"])
    assert {DEFAULT_CHECKPOINT, APACHE_CHECKPOINT} <= set(module.SETS["mapanything"])


def test_the_dockerfile_copies_files_that_exist():
    text = (ROOT / "docker" / "Dockerfile").read_text()
    copied = [
        line.split("COPY ", 1)[1].split()[0]
        for line in text.splitlines()
        if line.startswith("COPY ")
    ]
    assert copied, "Dockerfile has no COPY lines"
    for relative in copied:
        assert (ROOT / relative).exists(), f"Dockerfile copies {relative}, which is not in the repo"


def test_compose_services_match_the_runbook_words():
    compose = (ROOT / "docker" / "docker-compose.yml").read_text()
    for service in ("fetch", "qc", "extract", "test", "shell"):
        assert f"{service}:" in compose


def test_compose_and_the_shard_runner_read_the_same_weight_cache():
    # `fetch` runs through compose and the shards run through plain docker. If
    # they disagree about where weights land, the shards start with an empty
    # cache and HF_HUB_OFFLINE=1 turns that into an immediate failure on every
    # GPU at once, which is expensive to discover on a node.
    compose = (ROOT / "docker" / "docker-compose.yml").read_text()
    shards = (ROOT / "docker" / "run_shards.sh").read_text()

    assert "HF_CACHE_DIR" in compose and "HF_CACHE_DIR" in shards
    assert ".hf-cache" in compose and ".hf-cache" in shards
    assert "hf-cache:/cache/huggingface" not in compose, "named volume is invisible to run_shards.sh"


def test_the_image_reference_is_overridable_everywhere_it_appears():
    # A node that pulls from a registry has to be able to say so without
    # editing tracked files.
    compose = (ROOT / "docker" / "docker-compose.yml").read_text()
    shards = (ROOT / "docker" / "run_shards.sh").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert "${IMAGE:-proxy-extract:0.1.0}" in compose
    assert '"${IMAGE:-proxy-extract:0.1.0}"' in shards
    assert "REGISTRY" in makefile and "linux/amd64" in makefile


def test_ci_publishes_the_same_thing_the_makefile_describes():
    # CI is the only builder that a node's `docker pull` depends on, so it must
    # not drift from the Makefile humans read: same version, same platform,
    # same Dockerfile.
    workflow = (ROOT / ".github" / "workflows" / "image.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert "make -s version" in workflow, "CI should read the version, not restate it"
    assert "version:" in makefile, "Makefile must expose the `version` target CI calls"
    assert "linux/amd64" in workflow
    assert "docker/Dockerfile" in workflow
    # A build that is never exercised is a build that fails on a node instead.
    assert "pytest /opt/proxy-extract/tests" in workflow


def test_the_makefile_and_first_run_script_agree_on_the_demo_clip():
    makefile = (ROOT / "Makefile").read_text()
    first = (ROOT / "docker" / "first_run.sh").read_text()
    assert "26_trevor_seg_0004" in makefile
    assert "26_trevor_seg_0004" in first
    assert "coarse6" in makefile and "depth_anything" in makefile
    assert "coarse6" in first and "depth_anything" in first


def _load_gallery():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_gallery", ROOT / "experiments" / "export_gallery.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gallery_export_refuses_to_invent_missing_figures(tmp_path, monkeypatch):
    gallery = _load_gallery()

    monkeypatch.setattr(gallery, "SRC", tmp_path / "empty")
    monkeypatch.setattr(gallery, "DST", tmp_path / "out")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="figures missing"):
        gallery.main()


def test_gallery_export_copies_every_listed_figure(tmp_path, monkeypatch):
    gallery = _load_gallery()

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for name, _, _ in gallery.ITEMS:
        (src / name).write_bytes(b"png")
    monkeypatch.setattr(gallery, "SRC", src)
    monkeypatch.setattr(gallery, "DST", dst)

    gallery.main()

    assert (dst / "index.html").exists()
    for name, _, _ in gallery.ITEMS:
        assert (dst / name).read_bytes() == b"png"
    html = (dst / "index.html").read_text()
    assert "proxy-extract 可视化" in html
    assert all(name in html for name, _, _ in gallery.ITEMS)
