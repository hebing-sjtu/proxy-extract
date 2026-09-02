"""Where the models run and at what precision.

Both backends face the same two questions and have to answer them the same way,
because they share a GPU and a batch. Keeping the answers here rather than in
each backend is also what makes the precision choice reviewable: it is the
single largest throughput lever in the pipeline, and it is not free.

On the depth side the model runs at 504x504 in a window of one frame, which is
a small enough forward pass that an H200 spends most of it moving weights
rather than multiplying. Loading those weights as bfloat16 halves the traffic
and puts the matmuls on tensor cores, and bfloat16 rather than float16 because
it keeps float32's exponent range - depth is metres, spanning 0.1 to several
thousand, and a format that overflows partway up that range would fail in the
one place nothing checks.
"""

from __future__ import annotations

# Names `torch` also knows, so a caller can pass any of them through.
DTYPES = ("auto", "float32", "bfloat16", "float16")


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
