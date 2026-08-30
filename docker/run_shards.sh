#!/usr/bin/env bash
# Fan one extraction out across every GPU on the node, one container each.
#
#   ./docker/run_shards.sh /work/data/high /work/out 8
#
# Each worker gets --shard i/N so the clip list is partitioned with no overlap,
# plus --resume so re-running after a crash picks up only what is missing, and
# --keep-going so one bad clip does not take its shard's remaining clips down
# with it. Logs land in <out>/logs/shard-i.log.
#
# Deliberately one container per GPU rather than one container with all GPUs:
# the backends are single-device, and process isolation means a CUDA OOM kills
# one shard instead of the run.

set -euo pipefail

VIDEO_DIR="${1:?usage: run_shards.sh <video-dir-in-container> <out-dir-in-container> [n-gpus]}"
OUT_DIR="${2:?usage: run_shards.sh <video-dir-in-container> <out-dir-in-container> [n-gpus]}"
N_GPUS="${3:-$(nvidia-smi --list-gpus | wc -l)}"

IMAGE="${IMAGE:-proxy-extract:0.1.0}"
# Same names compose uses, so a server exports one set of variables and both
# entry points agree.
DATA_HOST="${DATA_DIR:-$(pwd)/handpick29_high_low}"
OUT_HOST="${OUT_DIR:-$(pwd)/out}"
# Must match docker-compose.yml's HF_CACHE_DIR default: `fetch` runs through
# compose, these shards run through plain docker, and they have to land on the
# same weights or every worker fails at once under HF_HUB_OFFLINE=1.
CACHE_HOST="${HF_CACHE_DIR:-$(pwd)/.hf-cache}"

SEMANTIC="${SEMANTIC:-coarse6}"
DEPTH="${DEPTH:-depth_anything}"

mkdir -p "$OUT_HOST/logs"

if [[ ! -d "$CACHE_HOST" ]] || [[ -z "$(ls -A "$CACHE_HOST" 2>/dev/null)" ]]; then
  echo "weight cache $CACHE_HOST is missing or empty." >&2
  echo "Run the fetch step first, or point HF_CACHE_DIR at a warm cache." >&2
  echo "These shards run offline; without weights all $N_GPUS would fail identically." >&2
  exit 1
fi

echo "image      $IMAGE"
echo "shards     $N_GPUS"
echo "videos     $VIDEO_DIR  (host: $DATA_HOST)"
echo "out        $OUT_DIR  (host: $OUT_HOST)"
echo "weights    $CACHE_HOST"
echo "backends   semantic=$SEMANTIC depth=$DEPTH"
echo

pids=()
for ((i = 0; i < N_GPUS; i++)); do
  docker run --rm \
    --gpus "device=$i" \
    --name "proxy-extract-shard-$i" \
    -v "$DATA_HOST:/work/data:ro" \
    -v "$OUT_HOST:/work/out" \
    -v "$CACHE_HOST:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" \
    proxy-extract extract \
      --video "$VIDEO_DIR" \
      --out "$OUT_DIR" \
      --semantic-backend "$SEMANTIC" \
      --depth-backend "$DEPTH" \
      --shard "$i/$N_GPUS" \
      --resume \
      --keep-going \
    >"$OUT_HOST/logs/shard-$i.log" 2>&1 &
  pids+=($!)
  echo "launched shard $i/$N_GPUS on GPU $i (pid ${pids[-1]})"
done

# Wait on each individually so one failing shard is named rather than hidden
# behind a bare non-zero exit.
failed=0
for ((i = 0; i < N_GPUS; i++)); do
  if ! wait "${pids[$i]}"; then
    echo "shard $i FAILED -- see $OUT_HOST/logs/shard-$i.log" >&2
    failed=1
  fi
done

if ((failed)); then
  echo "at least one shard failed; re-run this script to retry (--resume skips finished clips)" >&2
  exit 1
fi

echo "all $N_GPUS shards done -> $OUT_HOST"
