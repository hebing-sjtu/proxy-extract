# Base image note
# ---------------
# This starts from plain python:3.12-slim rather than an nvidia/cuda image.
# Since torch 2.x the linux/x86_64 wheel on PyPI ships its own CUDA runtime as
# `nvidia-*` pip packages, so the only thing the image needs from the host is
# the driver, which nvidia-container-toolkit injects at `docker run --gpus`
# time. Pulling a CUDA base as well would mean two copies of the runtime in the
# image and one more version pair to keep in sync for no gain.
#
# It follows that the host driver, not this file, decides which CUDA works.
# torch 2.13's default wheel wants driver >= 525. Check with `nvidia-smi`.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg: not strictly required (the opencv wheel bundles its own decoders) but
# it is what everyone reaches for to inspect a clip when a run looks wrong.
# libglib2.0-0: still dlopen-ed by opencv-python-headless at import.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Requirements before source so an edit to the package does not re-download
# ~3 GB of torch.
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY proxy-extract /opt/proxy-extract
RUN pip install --no-deps -e /opt/proxy-extract

# Weights live on a mounted volume, never baked in: they are gigabytes, several
# are gated, and re-pulling them on every image rebuild is the slowest possible
# way to iterate. `docker/fetch_models.py` fills this directory.
ENV HF_HOME=/cache/huggingface
# 777 so the image still works when run with `--user $(id -u)`, which is how you
# avoid root-owned files appearing in your working copy.
RUN mkdir -p /cache/huggingface && chmod 777 /cache/huggingface
# matplotlib writes a font cache on first import and warns loudly when $HOME is
# not writable, which it is not under `--user`.
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY docker/fetch_models.py /opt/fetch_models.py
COPY experiments /opt/experiments

WORKDIR /work
CMD ["proxy-extract", "--help"]
