#!/usr/bin/env bash
# Build the .venv this pipeline runs in, without touching anything else.
#
#   scripts/setup_venv.sh              # core + model backends + tests
#   VENV=/data/binghe/venvs/proxy scripts/setup_venv.sh
#   EXTRAS=core scripts/setup_venv.sh  # no torch, for contract/QC/encoding work
#
# A venv rather than the container: these nodes already carry a working main
# environment, and the image route wants a docker daemon, the container toolkit
# and root. A venv is the smaller intervention — it installs nothing outside its
# own directory and is deleted by removing that directory.
#
# Versions come from ../requirements.txt, the same file the image installs from,
# so a venv and a container cannot disagree about what RUNBOOK.md measured.

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

# ---------------------------------------------------------------------- verify

echo
echo "=== self-check ==="
"$py" -m pytest "$here/proxy-extract/tests" -q || die "the test suite does not pass in this venv"

"$py" - <<'PY'
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
    print(f"cuda available   {torch.cuda.is_available()} ({torch.cuda.device_count()} device(s))")
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

Next: pull the weights, then section 12 of RUNBOOK.md for the delivery run.

  $py scripts/fetch_models.py --set default
EOF
