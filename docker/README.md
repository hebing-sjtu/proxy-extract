# The image, in detail

For "how do I run this", read [`../RUNBOOK.md`](../RUNBOOK.md) first — the
one-shot path is `./docker/first_run.sh`. This file
is the parts you only need when something about the image itself is wrong.

## What is in it

| Layer | Contents |
| --- | --- |
| `python:3.12-slim` | Python and libc |
| apt | `ffmpeg`, `libglib2.0-0`, `git` |
| `requirements.txt` | torch, transformers, opencv, numpy, matplotlib, pytest — all pinned |
| `pip install -e` | `proxy-extract` at `/opt/proxy-extract`, giving the `proxy-extract` command |
| copies | `/opt/fetch_models.py`, `/opt/experiments` |

Model weights are **not** in the image. They live in a host directory mounted at
`/cache/huggingface`. See "Weights" below.

## Platform

The image must be `linux/amd64`. The torch wheel that bundles CUDA is only
published for linux/x86_64, so an arm64 build gets a CPU-only torch at best and
usually does not resolve at all. On an arm64 machine, never plain `docker
build`; use `make buildx`, which pins `--platform linux/amd64`.

## Publishing

`IMAGE` is overridable everywhere it appears — compose, `run_shards.sh`, and the
Makefile — so a node can run a pulled image without editing tracked files:

```bash
# on an amd64 builder
make buildx REGISTRY=registry.example.com/team    # cross-builds and pushes

# on each GPU node
make pull REGISTRY=registry.example.com/team
```

`make buildx` pushes when `REGISTRY` is set and loads into the local daemon when
it is not, because buildx cannot do both for a cross-architecture build.

Without a registry, `make save` / `make load` moves the image as a tarball.
Weights are not in it, so rsync `.hf-cache/` separately.

## Why not an `nvidia/cuda` base

Since torch 2.x, the linux/x86_64 wheel on PyPI bundles its own CUDA runtime as
`nvidia-*` pip dependencies. The only thing needed from the host is the driver,
which `nvidia-container-toolkit` injects when you pass `--gpus`. A CUDA base
image would add a second copy of the runtime and a version pair to keep in
sync, for nothing.

The practical consequence: **the host driver decides which CUDA you get**, not
this Dockerfile. `torch==2.13.0`'s default wheel needs driver >= 525. Check with
`nvidia-smi` on the host before blaming the image.

If you must target a specific CUDA build, change the torch lines in
`requirements.txt` to use the matching index, for example:

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.13.0+cu121
```

## Weights

The cache is a host directory, `./.hf-cache` by default and `HF_CACHE_DIR`
otherwise — deliberately not a named volume. `fetch` runs through compose while
the batch shards run through plain `docker run`, and both have to reach the same
weights. Since extraction forces `HF_HUB_OFFLINE=1`, a mismatch is not a slow
path but an immediate failure on every GPU at once, so `run_shards.sh` refuses
to start when the directory is empty.

`scripts/fetch_models.py` fills the cache. Groups, so you do not wait on
gigabytes you will not use:

| `--set` | Repos | Needed for |
| --- | --- | --- |
| `default` | `mask2former-swin-large-ade-semantic`, `Depth-Anything-V2-Metric-Outdoor-Small-hf` | `--semantic-backend coarse6 --depth-backend depth_anything` |
| `semantic` | the above Mask2Former, plus `segformer-b5-finetuned-cityscapes-1024-1024` | `ade20k`, `cityscapes` |
| `depth` | Depth Anything outdoor + indoor | `depth_anything` |
| `mapanything` | `facebook/map-anything`, `facebook/map-anything-apache` | `--depth-backend mapanything` |

`facebook/map-anything` is **gated and CC-BY-NC**: accept the terms on the hub
and `huggingface-cli login` first, and do not ship a commercial dataset built
from it. `facebook/map-anything-apache` is the Apache-2.0 variant and is the one
to use for anything shipped.

Once the cache is warm, run everything with `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`. This is not tidiness: 8 workers each hitting the hub on
start will rate-limit each other, and a hub hiccup mid-run leaves a
half-processed dataset.

Slow hub from mainland China: `export HF_ENDPOINT=https://hf-mirror.com` before
the fetch. Compose passes it through.

## Rebuilding

`requirements.txt` is copied and installed before the source, so editing the
package rebuilds in seconds. Editing `requirements.txt` re-downloads torch —
about 3 GB.

The package is installed with `-e`, so if you also bind-mount your working copy
over `/opt/proxy-extract` your edits take effect without a rebuild:

```bash
docker run --rm -v "$(pwd)/proxy-extract:/opt/proxy-extract" proxy-extract:0.1.0 pytest /opt/proxy-extract/tests -q
```

## Verifying the image

```bash
# 1. imports and CLI wiring, no GPU, no weights
docker run --rm proxy-extract:0.1.0 proxy-extract --help

# 2. the test suite, no GPU, no weights, ~15 s
docker run --rm proxy-extract:0.1.0 pytest /opt/proxy-extract/tests -q -p no:cacheprovider

# 3. the GPU is actually visible
docker run --rm --gpus all proxy-extract:0.1.0 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

If step 3 prints `False`, the image is fine and the host is not: check
`nvidia-smi`, then that `nvidia-container-toolkit` is installed and the Docker
daemon was restarted after installing it.

## Not verified

This image has not been built or run — the machine it was written on has no
Docker daemon. The pins come from a working local environment and the layout is
conventional, but treat the first `docker build` as something to watch rather
than something known to pass.
