#!/usr/bin/env bash
# Build the .venv this pipeline runs in, without touching anything else.
#
#   scripts/setup_venv.sh              # core + model backends + tests
#   DA3=1 scripts/setup_venv.sh        # also the default depth backend
#   VENV=/data/binghe/venvs/proxy scripts/setup_venv.sh
#   EXTRAS=core scripts/setup_venv.sh  # no torch, for contract/QC/encoding work
#
# A venv rather than the node's main environment: it installs nothing outside
# its own directory and is uninstalled by removing that directory, which is the
# smallest intervention on a machine that already carries a working environment.
#
# Versions come from ../requirements.txt, which is what RUNBOOK.md measured.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$here/.venv}"
EXTRAS="${EXTRAS:-full}"
PYTHON="${PYTHON:-python3}"

die() { echo "error: $*" >&2; exit 1; }

# ------------------------------------------------------------------ pre-flight

command -v "$PYTHON" >/dev/null || die "$PYTHON not found; set PYTHON=/path/to/python3"

# 3.10 is proxy-extract's floor. Checked before the venv is made, because the
# failure otherwise lands halfway through a multi-gigabyte torch download.
"$PYTHON" - <<'PY' || die "need Python >= 3.10"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

# Not fatal here: the venv is still worth building, and on a node without root
# the fix is a pip install into the venv we are about to make. Reported again
# after the install, by which time imageio-ffmpeg may have supplied one.
FFMPEG_MISSING=0
command -v ffmpeg >/dev/null || FFMPEG_MISSING=1

[[ -f "$here/requirements.txt" ]] || die "no requirements.txt at $here"

# The venv must not inherit the main environment's site-packages: that is the
# whole point of using one here.
if [[ -d "$VENV" ]]; then
  echo "reusing $VENV"
else
  echo "creating $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# `$VENV/bin/python -m pip` rather than `$VENV/bin/pip`: a venv moved or copied
# after creation keeps a stale shebang in the pip script, and the module form
# does not care.
py="$VENV/bin/python"
"$py" -m pip install --upgrade pip >/dev/null

# ------------------------------------------------------------------- install

case "$EXTRAS" in
  full)
    echo "installing pinned requirements (torch included; this is the slow part)"
    "$py" -m pip install -r "$here/requirements.txt"
    "$py" -m pip install --no-deps -e "$here/proxy-extract"
    ;;
  core)
    echo "installing core only: no torch, so the model backends stay unavailable"
    "$py" -m pip install -e "$here/proxy-extract[dev]"
    ;;
  *)
    die "EXTRAS must be 'full' or 'core', got '$EXTRAS'"
    ;;
esac

# depth-anything-3 is the default depth backend but cannot go in
# requirements.txt: it is not on PyPI, and it declares numpy<2 and
# python<=3.13, both of which contradict the pins above. Installed here with
# resolution switched off, which is safe only because the four packages it
# actually needs at runtime are named explicitly. Opt-in, since the weights are
# 6.8 GB and CC-BY-NC; without it, use DEPTH=depth_anything.
if [[ "${DA3:-0}" == "1" ]]; then
  echo "installing depth-anything-3 (--no-deps; see RUNBOOK section 5)"
  "$py" -m pip install --no-deps --ignore-requires-python \
    "git+https://github.com/ByteDance-Seed/depth-anything-3"
  "$py" -m pip install einops omegaconf addict imageio
fi

# ---------------------------------------------------------------------- verify

echo
echo "=== self-check ==="
"$py" -m pytest "$here/proxy-extract/tests" -q || die "the test suite does not pass in this venv"

"$py" - <<'PY'
import shutil

import proxy_extract
from proxy_extract.proxy import EncodeError, ffmpeg_binary

print(f"proxy_extract    {proxy_extract.__file__}")
try:
    print(f"ffmpeg           {ffmpeg_binary()}")
except EncodeError as error:
    print(f"ffmpeg           MISSING\n{error}")
try:
    import torch

    print(f"torch            {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"cuda available   {available} ({torch.cuda.device_count() if available else 0} device(s))")
    # Worth saying loudly here rather than leaving it to the first run: the
    # wheel PyPI serves for this version is built against CUDA 13, and a node
    # whose driver is 12.x gets no GPUs from it, quietly.
    if not available and shutil.which("nvidia-smi"):
        print(
            f"\nwarning: this node has nvidia-smi but torch cannot use it. torch was built\n"
            f"against CUDA {torch.version.cuda}; if the driver is older, install the matching\n"
            "build, e.g. for CUDA 12.6:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu126 \\\n"
            "      torch==2.13.0+cu126 torchvision==0.28.0+cu126"
        )
except ImportError:
    print("torch            not installed (EXTRAS=core)")
PY

if ((FFMPEG_MISSING)); then
  echo
  echo "note: no system ffmpeg was found, so the bundled imageio-ffmpeg build is being" >&2
  echo "      used instead. That is supported. Install a system one if you want the" >&2
  echo "      codecs your distro ships." >&2
fi

cat <<EOF

done. Activate with:

  source $VENV/bin/activate

or drive it without activating:

  $py -m proxy_extract --help

Next: pull the weights, then section 3 of RUNBOOK.md for the delivery run.

  $py scripts/fetch_models.py --set default
EOF
