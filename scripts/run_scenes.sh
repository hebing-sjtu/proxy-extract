#!/usr/bin/env bash
# Fan the 720p delivery extraction out across every GPU on this node.
#
#   scripts/run_scenes.sh
#   DATA_DIR=... OUT_DIR=... N_GPUS=4 scripts/run_scenes.sh
#
# One process per GPU, each with --shard i/N so the episode list is partitioned
# with no overlap, --resume so a re-run picks up only what is missing, and
# --keep-going so one unreadable episode costs one scene rather than its whole
# shard. Logs land in <out>/logs/shard-i.log.
#
# One process per GPU rather than one process with all of them: the backends are
# single-device, and process isolation means a CUDA OOM kills one shard instead
# of the run. It also caps host memory per process, which matters here - see the
# RAM pre-flight below.
#
# Bare metal rather than containers, because the H200 nodes are driven from a
# venv. The container route is in RUNBOOK.md section 3 and takes the same
# arguments.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data/binghe/datasets/ABot-World-Explorer-subset2000/data}"
OUT_DIR="${OUT_DIR:-/data/binghe/datasets/ABot-seg-long-2000}"
SEMANTIC="${SEMANTIC:-standard11}"
# depth_anything, not mapanything: this is the pair `fetch_models.py --set
# default` actually downloads, and its package ships in requirements.txt.
# mapanything is the better backend — multi-frame, so its scale is consistent
# across an episode — but it is not on PyPI and its weights are gated, so having
# it as the default meant following the runbook exactly could not work. Opt in
# with DEPTH=mapanything once it is installed; see RUNBOOK section 7.
DEPTH="${DEPTH:-depth_anything}"

# Default to the repo's own venv, which is the maintained deployment path, and
# fall back to whatever `python` is on PATH so an activated environment still
# works. Not `python3`: inside an activated venv that can resolve outside it.
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$_repo/.venv/bin/python" ]]; then
  PYTHON="$_repo/.venv/bin/python"
fi
PYTHON="${PYTHON:-python}"

# Measured: 465 MiB per 1800-frame episode, and ~40 GiB peak RSS per worker.
# Both are checked up front, because running out of either at episode 1200 of
# 2000 wastes far more than the thirty seconds this costs.
MIB_PER_SCENE="${MIB_PER_SCENE:-465}"
GIB_PER_WORKER="${GIB_PER_WORKER:-40}"

if [[ -z "${N_GPUS:-}" ]]; then
  N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$N_GPUS" -gt 0 ]] || { echo "no GPUs found; set N_GPUS=1 to run on CPU" >&2; exit 1; }
fi

die() { echo "error: $*" >&2; exit 1; }

# ------------------------------------------------------------------ pre-flight

$PYTHON -c 'import proxy_extract' 2>/dev/null \
  || die "proxy_extract is not importable by '$PYTHON'. Activate the venv, or: pip install -e proxy-extract"
# Asked of the code rather than of PATH, because it also accepts $FFMPEG and the
# static build imageio-ffmpeg vendors; `command -v ffmpeg` would reject a node
# that is in fact fine.
$PYTHON -c 'from proxy_extract.proxy import ffmpeg_binary; print(ffmpeg_binary())' >/dev/null 2>&1 \
  || die "$($PYTHON -c 'from proxy_extract.proxy import ffmpeg_binary
try:
    ffmpeg_binary()
except Exception as error:
    print(error)' 2>&1)"
[[ -d "$DATA_DIR" ]] || die "no such data directory: $DATA_DIR"

episodes="$(find "$DATA_DIR" -name video.mp4 -type f 2>/dev/null | wc -l | tr -d ' ')"
[[ "$episodes" -gt 0 ]] || die "no video.mp4 under $DATA_DIR (expected data/<prefix>/<sample_id>/video.mp4)"

mkdir -p "$OUT_DIR/logs"

# Space. `df` reports the filesystem the output lands on, which on these nodes
# is not the one the checkout is on.
need_mib=$((episodes * MIB_PER_SCENE))
avail_mib="$(df -Pm "$OUT_DIR" | awk 'NR==2 {print $4}')"
if [[ "$avail_mib" -lt "$need_mib" ]]; then
  die "$OUT_DIR has $((avail_mib / 1024)) GiB free but $episodes episodes need about $((need_mib / 1024)) GiB.
       Point OUT_DIR at a bigger filesystem, or deliver fewer episodes."
fi

# Host RAM. The temporal stages and the protagonist tracker run over the whole
# episode, so each worker holds the full 720p stacks; this is the number that
# decides how many workers fit, not the GPU.
if [[ -r /proc/meminfo ]]; then
  total_gib=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  want_gib=$((N_GPUS * GIB_PER_WORKER))
  if [[ "$total_gib" -lt "$want_gib" ]]; then
    fits=$((total_gib / GIB_PER_WORKER))
    echo "warning: $N_GPUS workers want about ${want_gib} GiB of host RAM but this node has ${total_gib} GiB." >&2
    echo "         About ${fits} workers fit. Set N_GPUS=${fits}, or expect the OOM killer." >&2
    echo >&2
  fi
fi

if [[ -n "${HF_HOME:-}" && ! -d "$HF_HOME" ]]; then
  echo "warning: HF_HOME=$HF_HOME does not exist; the first worker will download weights into it." >&2
fi

# The synthetic backends fabricate their predictions. They produce videos that
# are structurally indistinguishable from real ones, so a run started with them
# by accident is only caught by looking at the pixels. Refuse unless it is asked
# for in as many words.
for pair in "semantic=$SEMANTIC" "depth=$DEPTH"; do
  if [[ "${pair#*=}" == "synthetic" && "${ALLOW_SYNTHETIC:-0}" != "1" ]]; then
    die "$pair is a placeholder that invents its output; the scenes would look valid and be worthless.
       Use semantic=standard11 depth=depth_anything, or set ALLOW_SYNTHETIC=1 to dry-run the plumbing."
  fi
done

# Load both backends and run one tiny inference before committing to thousands
# of episodes. The heavy imports are lazy, inside the first call, so merely
# constructing a backend proves nothing — a missing package or an unfetched
# checkpoint would otherwise surface once per episode, for every episode.
echo "checking the backends load (first run also downloads weights)..."
$PYTHON - "$SEMANTIC" "$DEPTH" <<'PY' || die "the backends could not run; fix the above before launching $N_GPUS workers"
import os
import socket
import sys

import numpy as np

from proxy_extract.depth import get_backend as depth_backend
from proxy_extract.semantic import get_backend as semantic_backend

# A stalled weight download would otherwise hang here with no output and no GPU
# activity, which reads as a wedged job rather than a network problem. This is a
# per-read timeout, so a slow but progressing download still finishes.
socket.setdefaulttimeout(float(os.environ.get("PREFLIGHT_SOCKET_TIMEOUT", "180")))

semantic, depth = sys.argv[1], sys.argv[2]
frame = np.zeros((64, 64, 3), dtype=np.uint8)

try:
    result = depth_backend(depth).estimate([frame], cameras=None)
except Exception as error:
    print(f"depth backend '{depth}' failed: {type(error).__name__}: {error}", file=sys.stderr)
    if isinstance(error, ImportError) and depth == "mapanything":
        print(
            "mapanything is not on PyPI. Either install it:\n"
            "  pip install 'git+https://github.com/facebookresearch/map-anything'\n"
            "or use the backend the default weight set covers:\n"
            "  DEPTH=depth_anything",
            file=sys.stderr,
        )
    elif depth == "mapanything" and isinstance(error, (OSError, socket.timeout)):
        # Its DINOv2 backbone comes from dl.fbaipublicfiles.com via torch.hub,
        # which no HF mirror covers, so this is the failure most likely to look
        # like a wedged job. See RUNBOOK section 7.
        print(
            "mapanything pulls its DINOv2 backbone through torch.hub from\n"
            "dl.fbaipublicfiles.com, which HF_ENDPOINT does not mirror. Pre-populate\n"
            f"{__import__('torch').hub.get_dir()}/checkpoints from a machine with egress,\n"
            "or use DEPTH=depth_anything, whose weights are all on the hub.",
            file=sys.stderr,
        )
    raise SystemExit(1)

if not result.metric:
    print(f"depth backend '{depth}' returns up-to-scale depth; delivery needs metres", file=sys.stderr)
    raise SystemExit(1)

try:
    semantic_backend(semantic).segment([frame])
except Exception as error:
    print(f"semantic backend '{semantic}' failed: {type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"  ok: semantic={semantic} depth={depth} (metric)")
PY

gib() { awk -v m="$1" 'BEGIN {printf "%.1f", m / 1024}'; }

cat <<EOF
data       $DATA_DIR  ($episodes episodes)
out        $OUT_DIR  ($(gib "$avail_mib") GiB free, need ~$(gib "$need_mib") GiB)
shards     $N_GPUS
backends   semantic=$SEMANTIC depth=$DEPTH
weights    ${HF_HOME:-<default HF cache>}

EOF

# ---------------------------------------------------------------------- launch

pids=()
for ((i = 0; i < N_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="$i" \
  $PYTHON -m proxy_extract scenes \
    --video "$DATA_DIR" \
    --recursive \
    --out "$OUT_DIR" \
    --semantic-backend "$SEMANTIC" \
    --depth-backend "$DEPTH" \
    --shard "$i/$N_GPUS" \
    --resume \
    --keep-going \
    >"$OUT_DIR/logs/shard-$i.log" 2>&1 &
  # `$!` rather than `${pids[-1]}`: negative subscripts need bash 4.3, and the
  # Mac used for dry runs ships 3.2.
  pid=$!
  pids+=("$pid")
  echo "launched shard $i/$N_GPUS on GPU $i (pid $pid)"
done

echo
echo "follow one:   tail -f $OUT_DIR/logs/shard-0.log"
echo "check totals: $PYTHON -m proxy_extract scenes-audit --out $OUT_DIR"
echo

# Wait on each individually so a failing shard is named rather than hidden
# behind a bare non-zero exit.
failed=0
for ((i = 0; i < N_GPUS; i++)); do
  if ! wait "${pids[$i]}"; then
    echo "shard $i FAILED -- see $OUT_DIR/logs/shard-$i.log" >&2
    failed=1
  fi
done

# ----------------------------------------------------------------------- audit

echo
echo "=== audit ==="
$PYTHON -m proxy_extract scenes-audit \
  --out "$OUT_DIR" --report "$OUT_DIR/audit.json" || true

if ((failed)); then
  echo >&2
  echo "at least one shard failed. Re-run this script to retry: --resume skips" >&2
  echo "scenes whose four videos already hold every frame." >&2
  exit 1
fi
