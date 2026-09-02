"""Where the models run and at what precision.

Every backend faces the same two questions, and keeping the answers here rather
than in each one is what makes the precision choice reviewable: it is the
largest throughput lever in the pipeline, and it is not free.

bfloat16 rather than float16 wherever there is a choice, because it keeps
float32's exponent range. Depth is in metres, spanning 0.1 to several thousand,
and a format that overflows partway up that range would fail in the one place
nothing checks.

Only the semantic backend asks this module for a dtype. DA3 runs its own
autocast inside `inference()` and casts its inputs to float32 on the way in, so
its weights stay float32 and the question does not arise - see
`depth._resolve_weight_dtype`, which is where that surprise is written down.
"""

from __future__ import annotations

import os

# Names `torch` also knows, so a caller can pass any of them through.
DTYPES = ("auto", "float32", "bfloat16", "float16")

# How many threads this process may use for CPU work. Read by the CLI and by
# the encoders; set by run_scenes.sh, which knows how many workers share the
# node.
THREAD_VARIABLE = "PROXY_EXTRACT_THREADS"


def thread_budget() -> int:
    """Threads this worker may use, or 0 for "decide for yourself".

    Zero rather than the core count, so that a single interactive run keeps
    every library's own default and only a fanned-out run is capped.
    """
    try:
        return max(int(os.environ.get(THREAD_VARIABLE, "0")), 0)
    except ValueError:
        return 0


def limit_threads(threads: int | None = None) -> int:
    """Cap this process's CPU thread pools, and say what it settled on.

    Every library here sizes its pool to the whole machine, because each of
    them assumes it is the only thing running. Sixty-four workers on a 128-core
    node then ask for thousands of threads between them, and the machine spends
    its time context-switching: the symptom is every GPU at 0% with no error
    anywhere, which is a genuinely hard thing to diagnose from the outside.

    OpenCV is the one that matters most - Farneback optical flow is the largest
    CPU cost in the pipeline and it parallelises over the whole pool - but torch
    also runs CPU ops between forwards, and both honour this per process.

    OMP_NUM_THREADS has to be set before torch is imported to take full effect,
    which is the launcher's job; this is the backstop for a worker started by
    hand, and the only way to reach OpenCV, which does not read that variable.

    On macOS this call is accepted and ignored: those wheels parallelise with
    GCD, whose pool OpenCV does not control, so `getNumThreads` keeps reporting
    the core count. The deployment target is Linux, where it takes effect.
    """
    threads = thread_budget() if threads is None else threads
    if threads <= 0:
        return 0

    try:
        import cv2

        cv2.setNumThreads(threads)
    except ImportError:  # pragma: no cover - cv2 is a hard dependency
        pass
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    return threads


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> str:
    """Turn `auto` into a concrete dtype for this device.

    Reduced precision only where it is both fast and supported. CPU float16 is
    emulated and slower than float32, and MPS has its own gaps, so `auto` only
    means anything on CUDA - and there it prefers bfloat16, falling back to
    float32 on pre-Ampere cards that would emulate it rather than run it.
    """
    if requested not in DTYPES:
        raise ValueError(f"unknown dtype {requested!r}; expected one of {list(DTYPES)}")
    if requested != "auto":
        return requested
    if not device.startswith("cuda"):
        return "float32"

    import torch

    return "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
