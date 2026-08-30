from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
CWM_SRC = WORKSPACE / "code-world-model" / "src"
SAMPLE_DATA = WORKSPACE / "handpick29_high_low"

sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def cwm_duv():
    """The real code-world-model DUV loader, so the contract is checked against
    the consumer rather than against our own restatement of it."""
    if not CWM_SRC.is_dir():
        pytest.skip(f"code-world-model not present at {CWM_SRC}")
    sys.path.insert(0, str(CWM_SRC))
    duv = pytest.importorskip("cwm_h3_inference.duv")
    return duv


@pytest.fixture(scope="session")
def cwm_constants():
    if not CWM_SRC.is_dir():
        pytest.skip(f"code-world-model not present at {CWM_SRC}")
    sys.path.insert(0, str(CWM_SRC))
    return pytest.importorskip("cwm_h3_inference.constants")


@pytest.fixture(scope="session")
def sample_clip() -> Path:
    clip = SAMPLE_DATA / "low" / "26_trevor_seg_0004.mp4"
    if not clip.is_file():
        pytest.skip(f"sample clip not available at {clip}")
    return clip


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260830)
