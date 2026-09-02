from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
CWM_SRC = WORKSPACE / "code-world-model" / "src"

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
def sample_clip(tmp_path_factory) -> Path:
    """One episode's worth of moving footage, synthesised rather than shipped.

    The end-to-end tests need a decodable clip of exactly one window, at the
    size the pipeline decodes to. Generating it keeps the suite runnable on a
    bare checkout, and keeps a corpus nobody can redistribute out of the loop.
    Content is irrelevant here — the synthetic backends ignore it — but motion
    is not, because the stabilisation stages solve optical flow over it.
    """
    import cv2

    from proxy_extract import contract
    from proxy_extract.pipeline import WORK_HEIGHT, WORK_WIDTH

    path = tmp_path_factory.mktemp("corpus") / "video.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (WORK_WIDTH, WORK_HEIGHT)
    )
    rng = np.random.default_rng(20260902)
    for index in range(contract.WINDOW_FRAMES):
        frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), np.uint8)
        frame[: WORK_HEIGHT // 2, :, 2] = 200
        frame[WORK_HEIGHT // 2 :, :, 1] = 150
        drift = (index * 7) % 400
        frame[300:600, 200 + drift : 700 + drift] = 210
        frame += rng.integers(0, 10, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260830)
