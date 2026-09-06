#!/usr/bin/env bash
# Cut clips straight from the corpus, across every GPU on this node.
#
#   scripts/run_clip_episodes.sh
#   DATA_DIR=... CLIPS_DIR=... scripts/run_clip_episodes.sh
#   LIMIT=4 scripts/run_clip_episodes.sh          # prove a node on 4 episodes
#
# The one-pass route. run_scenes.sh delivers whole episodes and run_clips.sh
# then slices them, which predicts depth and semantics for every frame and
# keeps a third of them. This plans the windows first and predicts only those,
# at 1344x768 rather than by way of 720p. Same clips out the other end.
#
# Use it when the long delivered segments are not themselves wanted. If they
# are, run run_scenes.sh: cutting from a delivery you are producing anyway is
# free, and this would be a second pass over the same corpus.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data/binghe/datasets/ABot-World-Explorer-subset2000/data}"
CLIPS_DIR="${CLIPS_DIR:-/data/binghe/datasets/ABot-sub-2000-clips}"
SEMANTIC="${SEMANTIC:-standard11}"
DEPTH="${DEPTH:-depth_anything_v3}"

PER_SCENE="${PER_SCENE:-5}"
FRAMES="${FRAMES:-124}"
FPS="${FPS:-24}"
WORK_SIZE="${WORK_SIZE:-1344x768}"

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$_repo/.venv/bin/python" ]]; then
  PYTHON="$_repo/.venv/bin/python"
fi
PYTHON="${PYTHON:-python}"

# A clip's finished size, plus room for one `.work` directory per worker. The
# working set is transient - colour PNGs and the two arrays for 128 frames at
# 1344x768, about 580 MiB - but every worker holds one at once, so it is the
# worker count that decides how much headroom is needed rather than the
# episode count.
MIB_PER_CLIP="${MIB_PER_CLIP:-8}"
MIB_PER_WORKER_SCRATCH="${MIB_PER_WORKER_SCRATCH:-600}"

# Far lighter than the delivery run: a window is 128 frames, so the label stack
# `derive` holds is 132 MiB rather than 1.7 GiB, and what is left is mostly the
# model. Raise it while nvidia-smi shows the cards short of full.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"

if [[ -z "${N_GPUS:-}" ]]; then
  N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$N_GPUS" -gt 0 ]] || { echo "no GPUs found; set N_GPUS=1 to run on CPU" >&2; exit 1; }
fi
n_workers=$((N_GPUS * WORKERS_PER_GPU))

cores="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
if [[ -z "${THREADS_PER_WORKER:-}" ]]; then
  THREADS_PER_WORKER=$((cores / n_workers))
  ((THREADS_PER_WORKER < 1)) && THREADS_PER_WORKER=1
  ((THREADS_PER_WORKER > 4)) && THREADS_PER_WORKER=4
fi
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
export OPENCV_FOR_THREADS_NUM="$THREADS_PER_WORKER"
export PROXY_EXTRACT_THREADS="$THREADS_PER_WORKER"

die() { echo "error: $*" >&2; exit 1; }

# ------------------------------------------------------------------ pre-flight

$PYTHON -c 'import proxy_extract' 2>/dev/null \
  || die "proxy_extract is not importable by '$PYTHON'. Activate the venv, or: pip install -e proxy-extract"
$PYTHON -c 'from proxy_extract.proxy import ffmpeg_binary; ffmpeg_binary()' >/dev/null 2>&1 \
  || die "no usable ffmpeg; see RUNBOOK section 5"
[[ -d "$DATA_DIR" ]] || die "no such data directory: $DATA_DIR"

episodes="$(find "$DATA_DIR" -name video.mp4 -type f 2>/dev/null | wc -l | tr -d ' ')"
[[ "$episodes" -gt 0 ]] || die "no video.mp4 under $DATA_DIR"
if [[ -n "${LIMIT:-}" ]]; then
  ((LIMIT > 0)) || die "LIMIT must be a positive episode count, not '$LIMIT'"
  ((episodes = episodes < LIMIT ? episodes : LIMIT))
fi
clips=$((episodes * PER_SCENE))

for pair in "semantic=$SEMANTIC" "depth=$DEPTH"; do
  if [[ "${pair#*=}" == "synthetic" && "${ALLOW_SYNTHETIC:-0}" != "1" ]]; then
    die "$pair is a placeholder that invents its output; the clips would look valid and be worthless.
       Set ALLOW_SYNTHETIC=1 to dry-run the plumbing."
  fi
done

if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
  $PYTHON -c '
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False; every worker would run on CPU.", file=sys.stderr)
    print("Run scripts/doctor.py, which says which wheel this driver needs.", file=sys.stderr)
    print("Set ALLOW_CPU=1 to proceed anyway.", file=sys.stderr)
    raise SystemExit(1)
print(f"  ok: torch sees {torch.cuda.device_count()} GPU(s), CUDA {torch.version.cuda}")
' || die "torch cannot use this node's GPUs; fix that before launching $n_workers workers"
fi

mkdir -p "$CLIPS_DIR/logs"
need_mib=$((clips * MIB_PER_CLIP + n_workers * MIB_PER_WORKER_SCRATCH))
avail_mib="$(df -Pm "$CLIPS_DIR" | awk 'NR==2 {print $4}')"
if [[ "$avail_mib" -lt "$need_mib" ]]; then
  die "$CLIPS_DIR has $((avail_mib / 1024)) GiB free but $clips clips plus $n_workers working
       directories need about $((need_mib / 1024)) GiB."
fi

gib() { awk -v m="$1" 'BEGIN {printf "%.1f", m / 1024}'; }

cat <<EOF
data       $DATA_DIR ($episodes episodes${LIMIT:+, limited})
clips      $CLIPS_DIR ($clips clips, $(gib "$avail_mib") GiB free, need ~$(gib "$need_mib") GiB)
shape      $PER_SCENE x $FRAMES frames at $FPS fps, models at $WORK_SIZE
shards     $n_workers ($N_GPUS GPU(s) x $WORKERS_PER_GPU worker(s))
threads    $THREADS_PER_WORKER per worker, of $cores core(s)
backends   semantic=$SEMANTIC depth=$DEPTH

EOF

# ---------------------------------------------------------------------- launch

extra=()
for word in ${CLIP_ARGS:-}; do
  extra+=("$word")
done
for option in ${DEPTH_OPTIONS:-}; do
  extra+=(--depth-backend-option "$option")
done
for option in ${SEMANTIC_OPTIONS:-}; do
  extra+=(--semantic-backend-option "$option")
done
if [[ -n "${LIMIT:-}" ]]; then
  extra+=(--limit "$LIMIT")
fi

pids=()
for ((i = 0; i < n_workers; i++)); do
  gpu=$((i % N_GPUS))
  CUDA_VISIBLE_DEVICES="$gpu" \
  $PYTHON -u -m proxy_extract clip-episodes \
    --video "$DATA_DIR" \
    --recursive \
    --clips-out "$CLIPS_DIR" \
    --per-scene "$PER_SCENE" \
    --frames "$FRAMES" \
    --fps "$FPS" \
    --work-size "$WORK_SIZE" \
    --semantic-backend "$SEMANTIC" \
    --depth-backend "$DEPTH" \
    ${extra[@]+"${extra[@]}"} \
    --shard "$i/$n_workers" \
    --resume \
    --keep-going \
    >"$CLIPS_DIR/logs/shard-$i.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "launched shard $i/$n_workers on GPU $gpu (pid $pid)"
done

echo
echo "follow one:   tail -f $CLIPS_DIR/logs/shard-0.log"
echo "check totals: $PYTHON -m proxy_extract clips-audit --clips-out $CLIPS_DIR --frames $FRAMES"
echo

done_at_start="$(find "$CLIPS_DIR" -maxdepth 2 -name clip_report.json 2>/dev/null | wc -l | tr -d ' ')"

heartbeat() {
  local every="${HEARTBEAT_SECONDS:-60}"
  ((every > 0)) || return 0
  set +e
  while sleep "$every"; do
    local done_n alive=0 load=""
    done_n="$(find "$CLIPS_DIR" -maxdepth 2 -name clip_report.json 2>/dev/null | wc -l | tr -d ' ')"
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    if [[ -r /proc/loadavg ]]; then
      load="load $(awk '{print $1}' /proc/loadavg || true)"
    else
      load="load $(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}' || true)"
    fi
    printf '[%s] %s/%s clips (+%s this run), %s/%s alive, %s\n' \
      "$(date +%H:%M:%S)" "$done_n" "$clips" "$((done_n - done_at_start))" \
      "$alive" "$n_workers" "$load"
  done
}
heartbeat &
heartbeat_pid=$!
disown "$heartbeat_pid" 2>/dev/null || true
trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT

failed=0
for ((i = 0; i < n_workers; i++)); do
  if ! wait "${pids[$i]}"; then
    echo "shard $i FAILED -- see $CLIPS_DIR/logs/shard-$i.log" >&2
    failed=1
  fi
done
kill "$heartbeat_pid" 2>/dev/null || true

echo
echo "=== audit ==="
$PYTHON -m proxy_extract clips-audit \
  --clips-out "$CLIPS_DIR" --frames "$FRAMES" --report "$CLIPS_DIR/audit.json" || true

if ((failed)); then
  echo >&2
  echo "at least one shard failed. Re-run this script to retry: --resume skips clips" >&2
  echo "whose two videos already hold every frame." >&2
  exit 1
fi
