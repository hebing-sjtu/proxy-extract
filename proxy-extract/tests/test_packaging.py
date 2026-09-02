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


@needs_checkout
def test_scipy_stays_pinned_even_though_nothing_here_imports_it():
    """It looks unused, and removing it stops the semantic backend loading.

    Mask2Former builds its training loss in `__init__`, and that loss requires
    scipy at construction for its Hungarian matching. Nothing here trains, so a
    grep for `import scipy` across this package finds nothing and the pin reads
    like leftovers — which is exactly how it would get dropped.
    """
    requirements = (ROOT / "requirements.txt").read_text()

    assert "scipy==" in requirements
    assert "Mask2Former" in requirements, "the pin needs its reason next to it"


@needs_checkout
@pytest.mark.parametrize("doc", ["RUNBOOK.md", "DATA_F.md"])
def test_the_docs_describe_the_per_frame_layout(doc):
    """The frame directories are half the delivered bytes; both docs name them.

    Each stream has to appear next to the extension it is written with, since
    reading `depth` as an image or `color` as an array is the mistake the
    layout section exists to prevent. The docs draw the tree differently, so
    match `<stream>/<anything>.<ext>` rather than one spelling of the path.
    """
    import re

    from proxy_extract import frames

    text = (ROOT / doc).read_text()
    assert f"{frames.FRAMES_DIRNAME}/" in text

    # The staging stream is deleted before the scene is done, so it is the
    # code's business and not the reader's.
    extensions = dict.fromkeys(frames.ARRAY_STREAMS, "npy")
    extensions.update(dict.fromkeys(frames.IMAGE_STREAMS, "png"))
    for stream in frames.STREAMS:
        extension = extensions[stream]
        assert re.search(rf"{stream}/\S*\.{extension}", text), (
            f"{doc} does not show {stream} frames as .{extension}"
        )


@needs_checkout
def test_the_launcher_keeps_streams_the_cli_knows():
    """`--keep-frames` deletes things, so the launcher's default must parse."""
    from proxy_extract.cli import parse_kept_streams
    from proxy_extract.frames import STREAMS

    launcher = (ROOT / "scripts" / "run_scenes.sh").read_text()
    makefile = (ROOT / "Makefile").read_text()

    default = ",".join(STREAMS)
    assert f'KEEP_FRAMES="${{KEEP_FRAMES:-{default}}}"' in launcher
    assert f"KEEP_FRAMES ?= {default}" in makefile
    assert parse_kept_streams(default) == STREAMS
    assert "--keep-frames" in launcher


@needs_checkout
def test_the_launcher_budgets_the_memory_a_worker_actually_takes():
    """Both numbers decide whether a 2000-episode run survives; keep them paired.

    The RAM figure sets how many workers fit on a node and the space figure
    whether the output filesystem can hold the result. Each is measured, and
    each is quoted in the runbook next to how it was measured, so the launcher
    and the prose have to agree on it.
    """
    launcher = (ROOT / "scripts" / "run_scenes.sh").read_text()
    runbook = (ROOT / "RUNBOOK.md").read_text()

    assert 'GIB_PER_WORKER="${GIB_PER_WORKER:-11}"' in launcher
    assert "11 GiB" in runbook
    # The per-scene budget has to cover the frame directories, which are an
    # order of magnitude past the videos they encode.
    assert 'MIB_PER_SCENE="${MIB_PER_SCENE:-5200}"' in launcher
    assert "4.6 GiB" in runbook


@needs_checkout
def test_the_docs_state_the_fixed_cost_of_a_stored_frame():
    """depth and semantic frames are raw arrays, so their size is arithmetic."""
    import numpy as np

    from proxy_extract import frames
    from proxy_extract.delivery import DELIVERY_HEIGHT, DELIVERY_WIDTH

    text = (ROOT / "DATA_F.md").read_text()
    pixels = DELIVERY_WIDTH * DELIVERY_HEIGHT

    depth_mib = pixels * np.dtype(frames.DEPTH_DTYPE).itemsize / 2**20
    label_mib = pixels * np.dtype(frames.LABEL_DTYPE).itemsize / 2**20

    assert f"{depth_mib:.3f} MiB" in text
    assert f"{label_mib:.3f} MiB" in text
