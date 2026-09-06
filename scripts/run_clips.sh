#!/usr/bin/env bash
# Cut the delivered segments into short SFT clips, across every core on the node.
#
#   scripts/run_clips.sh
#   OUT_DIR=... CLIPS_DIR=... scripts/run_clips.sh
#   LIMIT=8 scripts/run_clips.sh                  # prove it on 8 segments first
#   PER_SCENE=5 FRAMES=124 FPS=24 scripts/run_clips.sh
#
# No GPU here, and that is the whole difference from run_scenes.sh. Cutting is
# reading PNGs and float16 arrays, two resizes and two x264 encodes: all CPU.
# So the shard count comes from the core count rather than the card count, and
# there is no backend to load, no weights to fetch and no VRAM to budget.
#
# Which also means this can run while the delivery run is still going. It only
# ever reads segments that `scenes-audit` calls complete, and `--resume` makes
# a second pass over a directory cheap, so the normal way to use it is to run
# it again each time another few hundred episodes land.

set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/binghe/datasets/ABot-seg-long-2000}"
CLIPS_DIR="${CLIPS_DIR:-${OUT_DIR}-clips}"

# 5 clips of 124 frames at 24 fps: one code-world-model window each, 5.17
# seconds of wall clock, evenly spread through the episode and never touching.
PER_SCENE="${PER_SCENE:-5}"
FRAMES="${FRAMES:-124}"
FPS="${FPS:-24}"

# 'frames' upscales the delivered 1280x720 colour, which keeps one DUV pixel
# exactly one 4x4 block of the target. 'source' re-decodes the 1920x1080
# original, which is sharper and gives that up - and needs the corpus mounted.
TARGET_FROM="${TARGET_FROM:-frames}"

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$_repo/.venv/bin/python" ]]; then
  PYTHON="$_repo/.venv/bin/python"
fi
PYTHON="${PYTHON:-python}"

# A clip is one x264 encode of 124 frames at 1344x768, one lossless 336x192
# encode, and a lossless PNG. The PNG is the largest single piece. This is an
# estimate and the check below is advisory: it is here to catch pointing
# CLIPS_DIR at a full disk, not to predict the total to a gigabyte.
MIB_PER_CLIP="${MIB_PER_CLIP:-8}"

# x264 wants about 1.5x the core count for one encode and each worker runs two,
# so without a cap the node asks for hundreds of threads and spends its time
# switching between them. Two each, which leaves the encodes threaded enough to
# keep up with the reads.
THREADS_PER_WORKER="${THREADS_PER_WORKER:-2}"

die() { echo "error: $*" >&2; exit 1; }

# ------------------------------------------------------------------ pre-flight

$PYTHON -c 'import proxy_extract' 2>/dev/null \
  || die "proxy_extract is not importable by '$PYTHON'. Activate the venv, or: pip install -e proxy-extract"
$PYTHON -c 'from proxy_extract.proxy import ffmpeg_binary; ffmpeg_binary()' >/dev/null 2>&1 \
  || die "no usable ffmpeg; see RUNBOOK section 5"
[[ -d "$OUT_DIR" ]] || die "no such delivery directory: $OUT_DIR"
[[ -f "$OUT_DIR/scenes_manifest.json" ]] \
  || die "no scenes_manifest.json under $OUT_DIR; that is written by the delivery run, so
       either OUT_DIR is wrong or nothing has been delivered here yet"

# Complete segments only, asked of the same code the cutter will ask. A segment
# still being written has frames that are about to change, and a clip cut from
# one is a snapshot of an intermediate state that nothing downstream can tell
# apart from a finished one.
segments="$($PYTHON -m proxy_extract scenes-audit --out "$OUT_DIR" --list complete | wc -l | tr -d ' ')"
[[ "$segments" -gt 0 ]] || die "no complete segments under $OUT_DIR yet; let the delivery run get further"

if [[ -n "${LIMIT:-}" ]]; then
  ((LIMIT > 0)) || die "LIMIT must be a positive segment count, not '$LIMIT'"
  ((segments = segments < LIMIT ? segments : LIMIT))
fi
clips=$((segments * PER_SCENE))

cores="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
if [[ -z "${N_WORKERS:-}" ]]; then
  N_WORKERS=$((cores / THREADS_PER_WORKER))
  ((N_WORKERS < 1)) && N_WORKERS=1
  # More shards than segments would launch workers with nothing to do, which is
  # harmless but reads as a hung shard in the logs.
  ((N_WORKERS > segments)) && N_WORKERS=$segments
fi

export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
export OPENCV_FOR_THREADS_NUM="$THREADS_PER_WORKER"
export PROXY_EXTRACT_THREADS="$THREADS_PER_WORKER"

mkdir -p "$CLIPS_DIR/logs"

need_mib=$((clips * MIB_PER_CLIP))
avail_mib="$(df -Pm "$CLIPS_DIR" | awk 'NR==2 {print $4}')"
if [[ "$avail_mib" -lt "$need_mib" ]]; then
  die "$CLIPS_DIR has $((avail_mib / 1024)) GiB free but $clips clips need about $((need_mib / 1024)) GiB.
       Point CLIPS_DIR at a bigger filesystem, or lower PER_SCENE."
fi

gib() { awk -v m="$1" 'BEGIN {printf "%.1f", m / 1024}'; }

cat <<EOF
segments   $OUT_DIR ($segments complete${LIMIT:+, limited})
clips      $CLIPS_DIR ($clips clips, $(gib "$avail_mib") GiB free, need ~$(gib "$need_mib") GiB)
shape      $PER_SCENE x $FRAMES frames at $FPS fps, target from $TARGET_FROM
shards     $N_WORKERS workers x $THREADS_PER_WORKER threads, of $cores core(s)

EOF

# ---------------------------------------------------------------------- launch

clips_args=()
for word in ${CLIPS_ARGS:-}; do
  clips_args+=("$word")
done
if [[ -n "${LIMIT:-}" ]]; then
  clips_args+=(--limit "$LIMIT")
fi

pids=()
for ((i = 0; i < N_WORKERS; i++)); do
  $PYTHON -u -m proxy_extract clips \
    --out "$OUT_DIR" \
    --clips-out "$CLIPS_DIR" \
    --per-scene "$PER_SCENE" \
    --frames "$FRAMES" \
    --fps "$FPS" \
    --target-from "$TARGET_FROM" \
    ${clips_args[@]+"${clips_args[@]}"} \
    --shard "$i/$N_WORKERS" \
    --resume \
    --keep-going \
    >"$CLIPS_DIR/logs/shard-$i.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "launched shard $i/$N_WORKERS (pid $pid)"
done

echo
echo "follow one:   tail -f $CLIPS_DIR/logs/shard-0.log"
echo "check totals: $PYTHON -m proxy_extract clips-audit --clips-out $CLIPS_DIR"
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
      "$alive" "$N_WORKERS" "$load"
  done
}
heartbeat &
heartbeat_pid=$!
disown "$heartbeat_pid" 2>/dev/null || true
trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT

failed=0
for ((i = 0; i < N_WORKERS; i++)); do
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
  echo "at least one shard failed. Re-run this script to retry: --resume skips" >&2
  echo "clips whose two videos already hold every frame." >&2
  exit 1
fi
