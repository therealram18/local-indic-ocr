#!/usr/bin/env bash
# Install IndicOCR's dependencies into the active Python environment.
#
#     ./install.sh                # pick the CUDA wheels from your driver
#     IDP_CUDA=cu121 ./install.sh # force a CUDA line
#     IDP_CUDA=cpu   ./install.sh # CPU only -- fine for layout, slow for the recognizer
#
# The one thing pip cannot work out for itself is which CUDA build you need. `pip install torch`
# takes the newest wheel, which on an older driver initialises to CPU with nothing but a warning --
# you get a working install that silently never touches the GPU. And torchvision pins an exact
# torch, so resolving the two separately is how you end up with a mismatched pair and a run of
# import errors. This installs both together, from one index.
set -euo pipefail

cd "$(dirname "$0")"
PY=${PYTHON:-python3}

# `uv venv` creates environments without pip, so `$PY -m pip` is not a safe assumption. Prefer
# whatever the environment actually has; uv is also just faster.
if command -v uv >/dev/null 2>&1; then
    pip_install() { uv pip install --python "$PY" "$@"; }
elif "$PY" -m pip --version >/dev/null 2>&1; then
    pip_install() { "$PY" -m pip install "$@"; }
elif "$PY" -m ensurepip --upgrade >/dev/null 2>&1; then
    pip_install() { "$PY" -m pip install "$@"; }
else
    echo "no installer found for $PY -- install pip (python -m ensurepip) or uv, then re-run" >&2
    exit 1
fi

detect_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo cpu; return; }
    local v
    v=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1/p' | head -1)
    case "$v" in
        # CUDA 12.x drivers run any cu12x wheel (minor version compatibility), so one build covers
        # the whole line. A 13.x wheel on a 12.x driver does NOT work -- that is the trap.
        12) echo cu124 ;;
        13|1[4-9]) echo cu130 ;;
        "") echo cpu ;;
        *)  echo cu124 ;;
    esac
}

CUDA=${IDP_CUDA:-$(detect_cuda)}
INDEX=https://download.pytorch.org/whl/$CUDA

echo "==> target: $CUDA"
[ "$CUDA" = cpu ] && echo "    (no GPU detected -- layout runs fine on CPU, the recognizer is slow)"

echo "==> torch + torchvision, together, from $INDEX"
pip_install torch torchvision --index-url "$INDEX"

echo "==> everything else"
pip_install -r requirements.txt

echo "==> checking"
$PY - <<'PY'
import torch, torchvision, transformers
print(f"    torch        {torch.__version__}")
print(f"    torchvision  {torchvision.__version__}")
print(f"    transformers {transformers.__version__}")
print(f"    cuda         {'yes, ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO -- running on CPU'}")
import sys; sys.path.insert(0, ".")
from indic_ocr import IndicOCR  # noqa: F401
print("    import       ok")
PY
echo "==> done"
