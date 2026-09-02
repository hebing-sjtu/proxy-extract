"""The launcher and the docs stay wired to the code they claim to drive.

A mismatch here is the kind that only shows up on a GPU node: the run starts,
`fetch_models` downloads the wrong checkpoint, and every worker then fails
identically hours later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# These assert things about the checkout — Makefile, launcher, docs — none of
# which are copied into an installed distribution. Run from an installed copy
# at /opt/proxy-extract, ROOT resolves to /opt and every one of them fails on a
# missing file rather than on a real mismatch.
needs_checkout = pytest.mark.skipif(
    not (ROOT / "scripts" / "run_scenes.sh").exists(),
    reason="packaging invariants describe the repository, not an installed copy",
)


def _load_script(name: str):
    """Load a `scripts/` module by path, because it is run as a script."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_checkout
def test_fetch_sets_use_the_same_repo_names_the_backends_do():
    from proxy_extract.depth.depth_anything import INDOOR_CHECKPOINT, OUTDOOR_CHECKPOINT
    from proxy_extract.depth.depth_anything_v3 import METRIC_CHECKPOINT, NESTED_CHECKPOINT
    from proxy_extract.depth.mapanything import APACHE_CHECKPOINT, DEFAULT_CHECKPOINT
    from proxy_extract.semantic.panoptic import ADE20K_CHECKPOINT, CITYSCAPES_CHECKPOINT

    module = _load_script("fetch_models")

    assert module.SETS["default"] == (ADE20K_CHECKPOINT, OUTDOOR_CHECKPOINT)
    assert ADE20K_CHECKPOINT in module.SETS["semantic"]
    assert CITYSCAPES_CHECKPOINT in module.SETS["semantic"]
    assert {INDOOR_CHECKPOINT, OUTDOOR_CHECKPOINT} <= set(module.SETS["depth"])
    assert {DEFAULT_CHECKPOINT, APACHE_CHECKPOINT} <= set(module.SETS["mapanything"])
    assert {NESTED_CHECKPOINT, METRIC_CHECKPOINT} <= set(
        module.SETS["da3"] + module.SETS["da3-apache"]
    )


@needs_checkout
def test_the_launcher_asks_for_backends_the_cli_accepts():
    """A typo here is only found by argparse, per shard, after the preflight."""
    from proxy_extract.cli import DEPTH_BACKENDS, SEMANTIC_BACKENDS

    launcher = (ROOT / "scripts" / "run_scenes.sh").read_text()

    assert 'SEMANTIC="${SEMANTIC:-standard11}"' in launcher
    assert "standard11" in SEMANTIC_BACKENDS
    assert 'DEPTH="${DEPTH:-depth_anything_v3}"' in launcher
    assert "depth_anything_v3" in DEPTH_BACKENDS


@needs_checkout
def test_the_launcher_and_the_makefile_agree_on_the_corpus_paths():
    launcher = (ROOT / "scripts" / "run_scenes.sh").read_text()
    makefile = (ROOT / "Makefile").read_text()

    for text in (launcher, makefile):
        assert "/data/binghe/datasets/ABot-World-Explorer-subset2000/data" in text
        assert "/data/binghe/datasets/ABot-seg-long-2000" in text


@needs_checkout
def test_the_makefile_only_names_targets_it_defines():
    """`make help` is the entry point, so a line it prints has to exist."""
    import re

    makefile = (ROOT / "Makefile").read_text()
    defined = set(re.findall(r"^([a-z][a-z0-9-]*):", makefile, re.MULTILINE))
    advertised = set(re.findall(r'@echo "  (make )?([a-z][a-z0-9-]*)\s', makefile))

    assert defined, "no targets found; the parse is wrong, not the Makefile"
    for _, target in advertised:
        assert target in defined, f"`make help` offers {target}, which is not a target"


@needs_checkout
@pytest.mark.parametrize("doc", ["RUNBOOK.md", "DATA_F.md"])
def test_the_docs_describe_the_layout_the_code_writes(doc):
    """The docs went stale against the code once; this is what noticed."""
    from proxy_extract import delivery

    text = (ROOT / doc).read_text()

    assert f"{delivery.SCENE_PREFIX}000000" in text
    assert f"{delivery.PROXY_DIRNAME}/" in text
    assert delivery.ANNOTATION_NAME in text
    for name in delivery.VIDEO_NAMES.values():
        assert name in text, f"{doc} does not mention {name}"


@needs_checkout
def test_the_docs_quote_the_encoding_constants_the_encoders_use():
    from proxy_extract import proxy

    text = (ROOT / "DATA_F.md").read_text()

    assert str(proxy.DEPTH_VIDEO_NEAR_METRES) in text
    assert str(int(proxy.DEPTH_VIDEO_FAR_METRES)) in text
    assert str(int(proxy.PROXY_FAR_METRES)) in text
    assert str(proxy.PROXY_SKY_CODE) in text
    assert "h264-logz-gray8" in text
    assert "libx264rgb" in text
