#!/usr/bin/env bash
# First-time walkthrough. Same five steps as RUNBOOK.md §2, just typed once.
#
#   ./docker/first_run.sh
#
# Needs: docker, an NVIDIA driver (>= 525), and handpick29_high_low/ in the
# repo root. Stops at the first failing step so you can see which one it was.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE=(docker compose -f docker/docker-compose.yml)
export DOCKER_UID
export DOCKER_GID
DOCKER_UID="$(id -u)"
DOCKER_GID="$(id -g)"

need() {
  command -v "$1" >/dev/null || {
    echo "missing $1 — install it before running this script" >&2
    exit 1
  }
}

need docker
need nvidia-smi

if [[ ! -d handpick29_high_low/high || ! -d handpick29_high_low/camera ]]; then
  echo "handpick29_high_low/{high,camera} not found under $ROOT" >&2
  exit 1
fi

mkdir -p out
# Must match docker-compose.yml's HF_CACHE_DIR default, and exist before the
# fetch step so the bind mount is owned by the invoking user rather than root.
mkdir -p .hf-cache

echo "==> 1/5  build"
"${COMPOSE[@]}" build

echo "==> 2/5  self-test (no GPU, no weights)"
"${COMPOSE[@]}" run --rm test

echo "==> 3/5  fetch weights"
"${COMPOSE[@]}" run --rm fetch

echo "==> 4/5  camera QC on the high renders"
"${COMPOSE[@]}" run --rm qc

echo "==> 5/5  extract + preview one clip"
"${COMPOSE[@]}" run --rm extract \
  extract --video /work/data/high/26_trevor_seg_0004.mp4 \
          --out /work/out/cond \
          --semantic-backend coarse6 \
          --depth-backend depth_anything
"${COMPOSE[@]}" run --rm extract \
  preview --condition-root /work/out/cond/high/26_trevor_seg_0004 \
          --out /work/out/preview.mp4

echo
echo "done."
echo "  QC report     out/camera_qc.json"
echo "  condition     out/cond/high/26_trevor_seg_0004/"
echo "  preview       out/preview.mp4"
echo "  measured figs gallery/index.html   (make gallery, no Docker needed)"
