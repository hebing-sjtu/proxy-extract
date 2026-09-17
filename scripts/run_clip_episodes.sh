#!/usr/bin/env bash
# Cut clips straight from the corpus, across every GPU on this node.
#
#   scripts/run_clip_episodes.sh
#   DATA_DIR=... CLIPS_DIR=... scripts/run_clip_episodes.sh
#   LIMIT=4 scripts/run_clip_episodes.sh          # prove a node on 4 episodes
#   NODE_COUNT=2 NODE_RANK=0 scripts/run_clip_episodes.sh   # and =1 on the other
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

die() { echo "error: $*" >&2; exit 1; }

DATA_DIR="${DATA_DIR:-/data/binghe/datasets/ABot-World-Explorer-subset2000/data}"
CLIPS_DIR="${CLIPS_DIR:-/data/binghe/datasets/ABot-sub-2000-clips}"
SEMANTIC="${SEMANTIC:-standard11}"
DEPTH="${DEPTH:-depth_anything_v3}"
# The consistency refiner, and PROXY_DUV_SPEC.md's per-frame form. Both default
# off because both cost real time; `REFINER=sam2 PROXY_DUV=1 DEPTH=moge3` is the
# configuration for a delivery that flickers. See RUNBOOK section 5.
REFINER="${REFINER:-none}"
PROXY_DUV="${PROXY_DUV:-0}"

PER_SCENE="${PER_SCENE:-5}"
FRAMES="${FRAMES:-124}"
FPS="${FPS:-24}"
WORK_SIZE="${WORK_SIZE:-1344x768}"

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# An activated environment wins over a repo-local .venv. That ordering matters
# inside the FastVideo Docker image, where the interpreter belongs to the image
# and a stale ./.venv left by an earlier attempt would otherwise silently take
# over - with a torch built for a different CUDA than the driver. See
# RUNBOOK_DOCKER.md.
if [[ -z "${PYTHON:-}" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
fi
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

# PROXY_DUV_SPEC.md's per-frame form is uncompressed, and it dwarfs everything
# else a clip holds: one float32 depth plane is 258,048 bytes, so 124 frames is
# about 31 MiB against the 8 MiB of the two videos put together. Left out of the
# estimate, the pre-flight would clear a 400 GiB run against an 80 GiB budget
# and the disk would fill somewhere in the middle of the corpus instead.
# Derived from FRAMES rather than hardcoded so it stays true if the shape moves.

# Far lighter than the delivery run: a window is 128 frames, so the label stack
# `derive` holds is 132 MiB rather than 1.7 GiB, and what is left is mostly the
# model. Raise it while nvidia-smi shows the cards short of full.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"

if [[ -z "${N_GPUS:-}" ]]; then
  N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$N_GPUS" -gt 0 ]] || { echo "no GPUs found; set N_GPUS=1 to run on CPU" >&2; exit 1; }
fi
n_workers=$((N_GPUS * WORKERS_PER_GPU))

# Several nodes over one corpus. `--shard i/N` partitions the episode list by
# position, and every worker derives that list itself from DATA_DIR, so the
# nodes need no coordination beyond agreeing on N and taking disjoint i. This
# node takes the block [NODE_RANK * n_workers, +n_workers).
#
# The arithmetic assumes **every node contributes the same worker count**. That
# is why it is checked rather than inferred: a node with 4 GPUs joining a run
# sized for 8 would silently leave half the corpus unclaimed, and the only
# symptom is a final audit that is short by episodes nobody looked at.
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
((NODE_COUNT >= 1)) || die "NODE_COUNT must be >= 1, got $NODE_COUNT"
((NODE_RANK >= 0 && NODE_RANK < NODE_COUNT)) \
  || die "NODE_RANK must be in [0, $NODE_COUNT), got $NODE_RANK"
n_shards=$((n_workers * NODE_COUNT))
shard_base=$((NODE_RANK * n_workers))

cores="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
if [[ -z "${THREADS_PER_WORKER:-}" ]]; then
  THREADS_PER_WORKER=$((cores / n_workers))
  ((THREADS_PER_WORKER < 1)) && THREADS_PER_WORKER=1
  ((THREADS_PER_WORKER > 4)) && THREADS_PER_WORKER=4
fi
# The other ceiling on WORKERS_PER_GPU, and the one with no error message: every
# worker decodes H.264 on the CPU to feed its GPU, so past one core each the
# workers queue for cores instead of running. The cards then look busier while
# finishing no faster, which reads as "there is still headroom" and invites
# raising the count again.
if ((cores < n_workers)); then
  echo "note: $n_workers workers on $cores cores - under one core each, so the decode" >&2
  echo "      side is now the limit and more workers will not finish sooner. GPU" >&2
  echo "      memory being free is not evidence to the contrary." >&2
  echo >&2
fi
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
export OPENCV_FOR_THREADS_NUM="$THREADS_PER_WORKER"
export PROXY_EXTRACT_THREADS="$THREADS_PER_WORKER"

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

# Are the selected backends actually importable? Both of the anti-flicker ones
# are git-only installs that a fresh node will not have, and without this the
# failure arrives as $n_workers identical ImportErrors in $n_workers separate
# log files, after the run has already been declared launched.
$PYTHON - "$DEPTH" "$REFINER" <<'PREFLIGHT' || die "a selected backend is not installed"
import importlib.util
import sys

depth, refiner = sys.argv[1], sys.argv[2]
needed = {
    "moge3": ("moge", "pip install git+https://github.com/microsoft/MoGe.git"),
    "sam2": ("sam2", "pip install git+https://github.com/facebookresearch/sam2.git"),
    "sam3": ("sam3", "pip install 'proxy-extract[sam3]'"),
}
missing = []
for choice in (depth, refiner):
    if choice in needed:
        module, how = needed[choice]
        if importlib.util.find_spec(module) is None:
            missing.append(f"  {choice} needs `{module}`, which is absent: {how}")
if missing:
    print("\n".join(missing), file=sys.stderr)
    raise SystemExit(1)
print(f"  ok: backends importable (depth={depth}, refiner={refiner})")
PREFLIGHT

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
mib_per_clip="$MIB_PER_CLIP"
if [[ "$PROXY_DUV" == "1" ]]; then
  # 258048 bytes of depth plus a compressible 8-bit id plane, per frame.
  mib_per_clip=$((mib_per_clip + FRAMES * 258048 / 1048576 + 1))
fi
need_mib=$((clips * mib_per_clip + n_workers * MIB_PER_WORKER_SCRATCH))
avail_mib="$(df -Pm "$CLIPS_DIR" | awk 'NR==2 {print $4}')"
if [[ "$avail_mib" -lt "$need_mib" ]]; then
  die "$CLIPS_DIR has $((avail_mib / 1024)) GiB free but $clips clips at ~$mib_per_clip MiB
       plus $n_workers working directories need about $((need_mib / 1024)) GiB.${PROXY_DUV:+
       PROXY_DUV=1 is most of that: the per-frame form is ~$((FRAMES * 258048 / 1048576)) MiB a clip, uncompressed.}"
fi

gib() { awk -v m="$1" 'BEGIN {printf "%.1f", m / 1024}'; }

# Host memory, which is what actually limits WORKERS_PER_GPU on this route and
# is easy to mistake for GPU memory. nvidia-smi showing the cards half empty
# invites raising the worker count, and the cards are not the constraint: one
# window is one batch - the depth backends lock the field of view and the metric
# scale per call, so it cannot be split - and that batch is resident in RAM
# three times over, once being worked on and twice prefetched, plus the float32
# depth stack derived from it.
#
#   colour   FRAMES+halo frames x WORK_SIZE x 3 bytes, x3 for the prefetch queue
#   depth    the same frame count as float32, x4 bytes
#   labels   the same again as uint8
#
# At 128 frames of 1344x768 that is about 1.2 GiB + 0.5 + 0.13, so the default
# allows 2 GiB a worker and rounds up for the models' host-side copies. An
# over-subscribed node does not fail cleanly either: it starts swapping, every
# worker slows together, and the GPUs go idle while the operator watches a
# throughput number fall for no visible reason.
MIB_PER_WORKER_RAM="${MIB_PER_WORKER_RAM:-2500}"

ram_need_mib=$((n_workers * MIB_PER_WORKER_RAM))
ram_avail_mib=""
if [[ -r /proc/meminfo ]]; then
  # MemAvailable rather than MemFree: the page cache is reclaimable, and on a
  # node that has just decoded a corpus MemFree reads near zero regardless.
  ram_avail_mib="$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)"
fi
if [[ -n "$ram_avail_mib" && "$ram_avail_mib" -lt "$ram_need_mib" ]]; then
  die "this node has $(gib "$ram_avail_mib") GiB of memory available but $n_workers workers
       need about $(gib "$ram_need_mib") GiB - one $((FRAMES + 4))-frame window each, held three
       times over for the prefetch, plus the depth stack.

       GPU memory is not the limit here; host memory is. Lower WORKERS_PER_GPU
       (currently $WORKERS_PER_GPU on $N_GPUS GPU(s)) to at most $((ram_avail_mib / MIB_PER_WORKER_RAM / N_GPUS)), or override the
       estimate with MIB_PER_WORKER_RAM if you have measured it on this node."
fi

cat <<EOF
data       $DATA_DIR ($episodes episodes${LIMIT:+, limited})
clips      $CLIPS_DIR ($clips clips at ~$mib_per_clip MiB, $(gib "$avail_mib") GiB free, need ~$(gib "$need_mib") GiB)
shape      $PER_SCENE x $FRAMES frames at $FPS fps, models at $WORK_SIZE
shards     $shard_base..$((shard_base + n_workers - 1)) of $n_shards ($N_GPUS GPU(s) x $WORKERS_PER_GPU worker(s), node $NODE_RANK of $NODE_COUNT)
threads    $THREADS_PER_WORKER per worker, of $cores core(s)
memory     ~$(gib "$ram_need_mib") GiB needed${ram_avail_mib:+, $(gib "$ram_avail_mib") GiB available}
backends   semantic=$SEMANTIC depth=$DEPTH refiner=$REFINER proxy_duv=$PROXY_DUV

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
if [[ "$PROXY_DUV" == "1" ]]; then
  extra+=(--proxy-duv)
fi

pids=()
for ((i = 0; i < n_workers; i++)); do
  gpu=$((i % N_GPUS))
  shard=$((shard_base + i))
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
    --refiner "$REFINER" \
    ${extra[@]+"${extra[@]}"} \
    --shard "$shard/$n_shards" \
    --resume \
    --keep-going \
    >"$CLIPS_DIR/logs/shard-$shard.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "launched shard $shard/$n_shards on GPU $gpu (pid $pid)"
done

echo
echo "follow one:   tail -f $CLIPS_DIR/logs/shard-$shard_base.log"
echo "check totals: $PYTHON -m proxy_extract clips-audit --clips-out $CLIPS_DIR --frames $FRAMES"
if [[ "$PROXY_DUV" == "1" ]]; then
  echo "spec checks:  $PYTHON -m proxy_extract proxy-duv-audit --root $CLIPS_DIR"
fi
echo

done_at_start="$(find "$CLIPS_DIR" -maxdepth 2 -name clip_report.json 2>/dev/null | wc -l | tr -d ' ')"

# Say so up front when the output root is not empty. The heartbeat counts every
# clip under it, so a root left over from an earlier run reads as though this
# one were nearly finished the moment it starts - and the question that matters
# is not how many clips are there but what made them. Resume reuses a clip only
# when it was cut with the same depth backend, refiner and --proxy-duv setting,
# so a run with different models redoes them rather than keeping a corpus that
# is half old predictions. It still costs the disk for both.
if ((done_at_start > 0)); then
  echo "note: $CLIPS_DIR already holds $done_at_start of $clips clips."
  echo "      Those were cut by an earlier run. This one reuses only the ones made"
  echo "      with DEPTH=$DEPTH, REFINER=$REFINER and PROXY_DUV=$PROXY_DUV; the rest"
  echo "      are cut again. For a clean corpus in its own directory instead:"
  echo "        CLIPS_DIR=/data/binghe/datasets/<new-name> $0"
  echo
fi

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
    printf '[%s] %s new this run, %s/%s clips on disk, %s/%s alive, %s\n' \
      "$(date +%H:%M:%S)" "$((done_n - done_at_start))" "$done_n" "$clips" \
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
    shard=$((shard_base + i))
    echo "shard $shard FAILED -- see $CLIPS_DIR/logs/shard-$shard.log" >&2
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
