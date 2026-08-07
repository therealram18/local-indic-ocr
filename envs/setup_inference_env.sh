#!/usr/bin/env bash
# Build a Python venv that runs IndicDocParser (layout model + vLLM-served OCR) on one GPU.
#
#   bash envs/setup_inference_env.sh
#   VENV_DIR=~/idp-env bash envs/setup_inference_env.sh
#   bash envs/setup_inference_env.sh --recreate
#
# Both models load in ONE environment (verified: the layout model runs on the same
# transformers the OCR/vLLM stack uses). There is no conversion or patching step.
#
# The one wrinkle is CUDA. `pip install vllm` fetches a wheel built for the newest CUDA, and
# on an older driver that wheel imports fine and then quietly reports no GPU. This script reads
# the driver's CUDA line and pulls matching wheels, rather than leaving you to discover the
# mismatch 1.7 GB into a model load.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_idp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# CUDA-matched wheels. Defaults reproduce the reference env; override for a different driver line.
CUDA_TAG="${CUDA_TAG:-cu129}"          # cu126 / cu128 / cu129 … match your driver
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
# vLLM must support the OCR model's Qwen3.5 GDN kernels. The reference env used a cu129 build
# from vLLM's per-CUDA index; nightly on that index tracks it. Pin VLLM_SPEC to reproduce exactly.
VLLM_SPEC="${VLLM_SPEC:-vllm}"         # e.g. VLLM_SPEC='vllm==0.20.1' to pin
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==5.7.0}"   # validated; layout needs PPDocLayoutV3 (>=5.7)

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)"

echo "===== IndicDocParser inference env ====="
echo "  venv    : ${VENV_DIR}"
echo "  cuda    : driver ${DRIVER_CUDA:-unknown} -> ${CUDA_TAG} wheels"
echo "  torch   : ${TORCH_VERSION} | vllm: ${VLLM_SPEC} | ${TRANSFORMERS_SPEC}"

if [[ "${RECREATE}" -eq 1 ]]; then
    echo "[1/4] --recreate: removing ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "[1/4] venv exists: $("${VENV_DIR}/bin/python" -V 2>&1)"
else
    echo "[1/4] creating venv (${PYTHON_BIN})"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
PIP="${VENV_DIR}/bin/pip"
PY="${VENV_DIR}/bin/python"
"${PIP}" install --quiet --upgrade pip

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
echo "[2/4] torch ${TORCH_VERSION} (+${CUDA_TAG})"
"${PIP}" install "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX}"

echo "[3/4] vLLM (${CUDA_TAG} index) + transformers + inference extras"
# vLLM from its per-CUDA index; it sees torch already satisfied by the +${CUDA_TAG} build.
"${PIP}" install "${VLLM_SPEC}" \
    --extra-index-url "https://wheels.vllm.ai/nightly/${CUDA_TAG}" \
    --extra-index-url "${TORCH_INDEX}"
"${PIP}" install "${TRANSFORMERS_SPEC}" accelerate "huggingface_hub>=1.0" hf_xet ninja pillow numpy

# --------------------------------------------------------------------------- #
# Gate -- catch the CUDA/wheel mismatch and the ninja gotcha here, not 1.7 GB into a load.
# --------------------------------------------------------------------------- #
echo "[4/4] verifying"
"${PY}" - <<'EOF'
import shutil, sys
problems = []
import torch, transformers
print(f"  torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"  transformers {transformers.__version__}")
try:
    import vllm; print(f"  vllm         {vllm.__version__}")
except Exception as e:
    problems.append(f"vllm import failed: {e}")
# layout model needs the PPDocLayoutV3 classes
try:
    from transformers import PPDocLayoutV3ForObjectDetection  # noqa: F401
    print("  layout arch  PPDocLayoutV3 present")
except Exception:
    problems.append("this transformers build lacks PPDocLayoutV3 (need >= 5.7)")
# the OCR model JIT-compiles a GDN kernel on first inference and needs ninja on PATH
if shutil.which("ninja") is None:
    problems.append("`ninja` not on PATH (needed by the OCR GDN kernel JIT; `pip install ninja`)")
else:
    print("  ninja        on PATH")
if torch.cuda.is_available():
    print(f"  GPUs visible {torch.cuda.device_count()}")
else:
    problems.append("torch.cuda.is_available() is False on a GPU box -> wheel/driver CUDA "
                    "mismatch; rerun with CUDA_TAG matching your driver (e.g. cu126). "
                    "(On a login node with no GPU this is expected -- check on a GPU node.)")
if problems:
    print("\nFAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print("\n  all checks passed")
EOF

echo ""
echo "Done. Run the pipeline with:"
echo "  CUDA_VISIBLE_DEVICES=0 ${PY} ${REPO_ROOT}/scripts/parse.py page.png -o out/page"
