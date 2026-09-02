#!/usr/bin/env bash
# Fan the 720p delivery extraction out across every GPU on this node.
#
#   scripts/run_scenes.sh
#   DATA_DIR=... OUT_DIR=... N_GPUS=4 scripts/run_scenes.sh
#   WORKERS_PER_GPU=8 scripts/run_scenes.sh   # fill the GPU's idle CPU phases
#   KEEP_FRAMES=depth scripts/run_scenes.sh   # a third of the space
#   SCENES_ARGS="--flow-downscale 4" scripts/run_scenes.sh   # cheaper flow
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
# Bare metal rather than containers: the H200 nodes are driven from a venv, so
# nothing here needs a docker daemon or root.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data/binghe/datasets/ABot-World-Explorer-subset2000/data}"
OUT_DIR="${OUT_DIR:-/data/binghe/datasets/ABot-seg-long-2000}"
SEMANTIC="${SEMANTIC:-standard11}"
# depth_anything_v3, because it is the only one of the three that both reports
# metres and can be obtained on a node whose egress stops at the hub. mapanything
# fetches its DINOv2 backbone through torch.hub from dl.fbaipublicfiles.com and
# hangs silently where that is blocked; depth_anything (V2) installs cleanly but
# sees one frame at a time. DA3 carries its backbone inside its own checkpoint.
#
# It is not pip-installable either — see RUNBOOK section 5 for the --no-deps
# recipe — so DEPTH=depth_anything remains the fallback that needs no extra
# install. The weights are CC-BY-NC 4.0: research use only.
DEPTH="${DEPTH:-depth_anything_v3}"

# Default to the repo's own venv, which is the maintained deployment path, and
# fall back to whatever `python` is on PATH so an activated environment still
# works. Not `python3`: inside an activated venv that can resolve outside it.
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$_repo/.venv/bin/python" ]]; then
  PYTHON="$_repo/.venv/bin/python"
fi
PYTHON="${PYTHON:-python}"

# Both are checked up front, because running out of either at episode 1200 of
# 2000 wastes far more than the thirty seconds this costs.
#
# Space: the per-frame directories dominate, and two of the four are a fixed
# size because they are raw arrays - at 1280x720, depth is 1.758 MiB a frame
# and semantic 0.879, so 4.6 GiB per 1800-frame episode before the two image
# streams. The videos are noise beside that, about 25 MiB an episode. Lower
# this together with KEEP_FRAMES: dropping to depth alone is ~3.2 GiB.
#
# Memory: measured 6.6 GiB flat through the streaming inference stage, plus
# about 2.4 MiB per frame while the protagonist tracker holds the label stack,
# so ~11 GiB for a 1800-frame episode. It used to be ~40, which is why one
# worker per GPU was the old default and six is this one.
MIB_PER_SCENE="${MIB_PER_SCENE:-5200}"
GIB_PER_WORKER="${GIB_PER_WORKER:-11}"

# Which per-frame streams to keep. Only depth holds something the videos do not
# - depth.mp4 quantises float16 metres onto 8 bits - so KEEP_FRAMES=depth keeps
# the informative half at a third of the space, and KEEP_FRAMES=none delivers
# videos alone. See RUNBOOK section 4.
KEEP_FRAMES="${KEEP_FRAMES:-color,depth,semantic,duv}"

if [[ -z "${N_GPUS:-}" ]]; then
  N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$N_GPUS" -gt 0 ]] || { echo "no GPUs found; set N_GPUS=1 to run on CPU" >&2; exit 1; }
fi

# Only part of an episode is GPU work. Decoding, the optical-flow stabilisation,
# the PNG writes and the ffmpeg encodes are all CPU, and DA3 sees one frame per
# call — its `window` is a multi-view setting, not a batch size, so raising it
# binds those frames' depth scales together rather than making the pass wider.
# A single worker therefore leaves its GPU idle for long stretches. Stacking
# workers on a card is what fills those gaps, with another worker's forward.
#
# Host RAM still bounds this rather than VRAM, but far less tightly than it did:
# a worker peaks near 11 GiB rather than 40, so six per GPU on eight cards wants
# roughly 530 GiB. Raise it while `nvidia-smi` shows the cards short of full and
# the RAM check below stays quiet.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-6}"
n_workers=$((N_GPUS * WORKERS_PER_GPU))

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

# Host RAM. Inference streams, but the protagonist tracker still has to see
# every frame before it can name the protagonist, so each worker holds one
# episode of labels; this is the number that decides how many workers fit.
if [[ -r /proc/meminfo ]]; then
  total_gib=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  want_gib=$((n_workers * GIB_PER_WORKER))
  if [[ "$total_gib" -lt "$want_gib" ]]; then
    fits=$((total_gib / GIB_PER_WORKER))
    echo "warning: $n_workers workers want about ${want_gib} GiB of host RAM but this node has ${total_gib} GiB." >&2
    echo "         About ${fits} workers fit. Lower N_GPUS or WORKERS_PER_GPU, or expect the OOM killer." >&2
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

# nvidia-smi listing GPUs is not the same as torch being able to use them: a
# wheel built for a newer CUDA than the driver leaves torch on CPU, with only a
# UserWarning to say so. N_GPUS then counts cards nobody can reach and the run
# proceeds at CPU speed on every shard, which on this workload means it never
# finishes. Checked against N_GPUS rather than unconditionally, so a deliberate
# CPU run still works by saying N_GPUS=1 ALLOW_CPU=1.
if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
  $PYTHON - "$N_GPUS" <<'PY' || die "torch cannot use this node's GPUs; fix that before launching $N_GPUS workers"
import sys

import torch

wanted = int(sys.argv[1])
try:
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
except Exception as error:
    available, count = False, 0
    print(f"torch.cuda raised: {type(error).__name__}: {error}", file=sys.stderr)

def driver_cuda() -> str:
    """The highest CUDA the installed driver can run, as nvidia-smi reports it.

    Read from the driver rather than from the toolkit, because there need not
    be a toolkit: the wheels bundle their own runtime, so the driver is the
    only thing that constrains which wheel works.
    """
    import re
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    found = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return found.group(1) if found else ""


if not available:
    driver = driver_cuda()
    # Within a major family any minor works: a cu12x wheel runs on any 12.x
    # driver from 525.60.13 up. Across families nothing does, which is the
    # usual version of this failure - a cu130 wheel on a 12.x driver.
    wheels = {"12": "cu126", "13": "cu130"}
    tag = wheels.get(driver.split(".")[0], "")
    version = torch.__version__.split("+")[0]

    print(
        f"nvidia-smi reports GPUs but torch.cuda.is_available() is False, so all {wanted}\n"
        f"workers would run on CPU. torch {torch.__version__} was built against CUDA\n"
        f"{torch.version.cuda}, and this node's driver goes up to CUDA {driver or '?'}.",
        file=sys.stderr,
    )
    if tag:
        try:
            import torchvision

            vision = f" torchvision=={torchvision.__version__.split('+')[0]}+{tag}"
        except ImportError:
            vision = ""
        print(
            f"\nInstall the {tag} build of the same versions:\n"
            f"  pip install --index-url https://download.pytorch.org/whl/{tag} \\\n"
            f"      torch=={version}+{tag}{vision}\n"
            f"\nThe +{tag} suffix matters: plain torch=={version} is already satisfied by\n"
            "the build that is failing here, so pip would do nothing.",
            file=sys.stderr,
        )
    print("\nSet ALLOW_CPU=1 to proceed anyway.", file=sys.stderr)
    raise SystemExit(1)

if count < wanted:
    print(f"warning: asked for {wanted} shards but torch sees {count} GPUs", file=sys.stderr)
print(f"  ok: torch sees {count} GPU(s), CUDA {torch.version.cuda}")
PY
fi

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
    if isinstance(error, ImportError) and depth == "depth_anything_v3":
        print(
            "depth-anything-3 is not on PyPI, and its declared pins (numpy<2,\n"
            "python<=3.13) fight this environment, so install it without them:\n"
            "  pip install --no-deps --ignore-requires-python \\\n"
            "      git+https://github.com/ByteDance-Seed/depth-anything-3\n"
            "  pip install einops omegaconf addict imageio\n"
            "Then fetch the 6.8 GB checkpoint:\n"
            "  python scripts/fetch_models.py --set da3\n"
            "Or fall back to the backend requirements.txt already covers:\n"
            "  DEPTH=depth_anything",
            file=sys.stderr,
        )
    elif isinstance(error, ImportError) and depth == "mapanything":
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
        # like a wedged job. See RUNBOOK section 5.
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
shards     $n_workers ($N_GPUS GPU(s) x $WORKERS_PER_GPU worker(s), ~$((n_workers * GIB_PER_WORKER)) GiB RAM)
backends   semantic=$SEMANTIC depth=$DEPTH
frames     $KEEP_FRAMES
weights    ${HF_HOME:-<default HF cache>}
extra      ${SCENES_ARGS:-<none>}${DEPTH_OPTIONS:+ }${DEPTH_OPTIONS:-}

EOF

# ---------------------------------------------------------------------- launch

# Space-separated KEY=VALUE, e.g. DEPTH_OPTIONS="process_res=728 window=1".
# Not dtype: DA3 runs its own bfloat16 autocast, so there is nothing to set.
depth_options=()
for option in ${DEPTH_OPTIONS:-}; do
  depth_options+=(--depth-backend-option "$option")
done

# Likewise for the semantic model, whose batch size is the one knob that widens
# its forward pass: SEMANTIC_OPTIONS="batch_size=8".
semantic_options=()
for option in ${SEMANTIC_OPTIONS:-}; do
  semantic_options+=(--semantic-backend-option "$option")
done

# Anything else to hand `scenes`, split on spaces, e.g. the stabilisation
# tradeoffs from RUNBOOK section 6:
#
#   SCENES_ARGS="--flow-downscale 4"
#   SCENES_ARGS="--flow-downscale 4 --temporal-radius 1"
#
# A passthrough rather than one variable per flag, so the launcher does not have
# to grow a mirror of the CLI. It is unvalidated on purpose: a typo reaches
# argparse, which refuses it by name in the shard log.
scenes_args=()
for word in ${SCENES_ARGS:-}; do
  scenes_args+=("$word")
done

pids=()
for ((i = 0; i < n_workers; i++)); do
  gpu=$((i % N_GPUS))
  CUDA_VISIBLE_DEVICES="$gpu" \
  $PYTHON -m proxy_extract scenes \
    --video "$DATA_DIR" \
    --recursive \
    --out "$OUT_DIR" \
    --semantic-backend "$SEMANTIC" \
    --depth-backend "$DEPTH" \
    --keep-frames "$KEEP_FRAMES" \
    ${depth_options[@]+"${depth_options[@]}"} \
    ${semantic_options[@]+"${semantic_options[@]}"} \
    ${scenes_args[@]+"${scenes_args[@]}"} \
    --shard "$i/$n_workers" \
    --resume \
    --keep-going \
    >"$OUT_DIR/logs/shard-$i.log" 2>&1 &
  # `$!` rather than `${pids[-1]}`: negative subscripts need bash 4.3, and the
  # Mac used for dry runs ships 3.2.
  pid=$!
  pids+=("$pid")
  echo "launched shard $i/$n_workers on GPU $gpu (pid $pid)"
done

echo
echo "follow one:   tail -f $OUT_DIR/logs/shard-0.log"
echo "check totals: $PYTHON -m proxy_extract scenes-audit --out $OUT_DIR"
echo

# Wait on each individually so a failing shard is named rather than hidden
# behind a bare non-zero exit.
failed=0
for ((i = 0; i < n_workers; i++)); do
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
