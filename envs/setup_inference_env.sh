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
# Uses `uv` if present (fast, and needs no system `ensurepip`); otherwise `python -m venv`.
# The one wrinkle is CUDA: `pip install vllm` fetches a wheel built for the newest CUDA, and on
# an older driver that wheel imports fine and then quietly reports no GPU. This script reads the
# driver's CUDA line and pulls matching wheels, and its final gate catches the mismatch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_idp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV="${UV:-$(command -v uv 2>/dev/null || echo "${HOME}/.local/bin/uv")}"

# CUDA-matched wheels. Defaults reproduce the reference env; override for a different driver line.
CUDA_TAG="${CUDA_TAG:-cu129}"          # cu126 / cu128 / cu129 … match your driver
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
# vLLM must support the OCR model's Qwen3.5 GDN kernels. The reference build came from vLLM's
# per-CUDA nightly index; pin VLLM_SPEC to reproduce exactly (e.g. VLLM_SPEC='vllm==0.20.1').
VLLM_SPEC="${VLLM_SPEC:-vllm}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==5.7.0}"   # validated; layout needs PPDocLayoutV3 (>=5.7)

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
VLLM_INDEX="https://wheels.vllm.ai/nightly/${CUDA_TAG}"

echo "===== IndicDocParser inference env ====="
echo "  venv    : ${VENV_DIR}"
echo "  cuda    : driver ${DRIVER_CUDA:-unknown} -> ${CUDA_TAG} wheels"
echo "  stack   : vllm=${VLLM_SPEC} (brings matched torch) | transformers>=5.7"
[[ -x "${UV}" ]] && echo "  builder : uv ($(${UV} --version 2>/dev/null))" || echo "  builder : python -m venv"

[[ "${RECREATE}" -eq 1 ]] && { echo "[1/4] --recreate: removing ${VENV_DIR}"; rm -rf "${VENV_DIR}"; }

# --------------------------------------------------------------------------- #
# Create the venv and define pipc() = "install into this venv"
# --------------------------------------------------------------------------- #
if [[ -x "${UV}" ]]; then
    [[ -x "${VENV_DIR}/bin/python" ]] || { echo "[1/4] uv venv (python ${PYTHON_VERSION})"; "${UV}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"; }
    pipc() { "${UV}" pip install --python "${VENV_DIR}/bin/python" "$@"; }
    # vLLM's per-CUDA wheels are pre-releases and its deps span several indexes.
    RESOLVE_FLAGS=(--prerelease=allow --index-strategy unsafe-best-match)
else
    [[ -x "${VENV_DIR}/bin/python" ]] || { echo "[1/4] python -m venv"; python3 -m venv "${VENV_DIR}"; }
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip
    pipc() { "${VENV_DIR}/bin/pip" install "$@"; }
    RESOLVE_FLAGS=(--pre)
fi
PY="${VENV_DIR}/bin/python"

# Install vLLM FIRST and let it pull its own matched torch from the CUDA index. Pre-pinning torch
# fails: a nightly vLLM pins an exact torch (e.g. 2.13.0) and will uninstall a different one,
# leaving a half-installed torch. vLLM (+ flashinfer/packaging deps) resolves across all indexes.
echo "[2/4] vLLM (${CUDA_TAG} index) — brings its matched torch"
pipc "${RESOLVE_FLAGS[@]}" "${VLLM_SPEC}" \
    --extra-index-url "${VLLM_INDEX}" \
    --extra-index-url "${TORCH_INDEX}" \
    --extra-index-url "https://pypi.org/simple"

# transformers: keep whatever vLLM pinned if it already satisfies the layout floor (>=5.7, which
# ships PPDocLayoutV3); otherwise install the floor. Then the inference extras.
echo "[3/4] transformers (>=5.7 for PPDocLayoutV3) + inference extras"
"${PY}" -c 'import transformers,sys; v=tuple(int(x) for x in transformers.__version__.split(".")[:2]); sys.exit(0 if v>=(5,7) else 1)' 2>/dev/null \
    || pipc "${RESOLVE_FLAGS[@]}" "transformers>=5.7" --extra-index-url "https://pypi.org/simple"
pipc accelerate "huggingface_hub>=1.0" hf_xet ninja pillow numpy scipy --extra-index-url "https://pypi.org/simple"

# --------------------------------------------------------------------------- #
# Gate -- catch the CUDA/wheel mismatch and the ninja gotcha here, not 1.7 GB into a load.
# --------------------------------------------------------------------------- #
echo "[4/4] verifying"
"${PY}" - <<'EOF'
import os, shutil, sys
# ninja lives in the venv's bin; the pipeline adds it to PATH at runtime, so mirror that here.
os.environ["PATH"] = os.path.join(sys.prefix, "bin") + os.pathsep + os.environ.get("PATH", "")
problems = []
import torch, transformers
print(f"  torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"  transformers {transformers.__version__}")
try:
    import vllm; print(f"  vllm         {vllm.__version__}")
except Exception as e:
    problems.append(f"vllm import failed: {e}")
try:
    from transformers import PPDocLayoutV3ForObjectDetection  # noqa: F401
    print("  layout arch  PPDocLayoutV3 present")
except Exception:
    problems.append("this transformers build lacks PPDocLayoutV3 (need >= 5.7)")
if shutil.which("ninja") is None:
    problems.append("`ninja` not on PATH (needed by the OCR GDN kernel JIT)")
else:
    print("  ninja        on PATH")
if torch.cuda.is_available():
    print(f"  GPUs visible {torch.cuda.device_count()}")
else:
    problems.append("torch.cuda.is_available() is False on a GPU box -> wheel/driver CUDA "
                    "mismatch; rerun with CUDA_TAG matching your driver (e.g. cu126). "
                    "(On a login node with no GPU this is expected.)")
if problems:
    print("\nFAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print("\n  all checks passed")
EOF

echo ""
echo "Done. Run the pipeline with:"
echo "  CUDA_VISIBLE_DEVICES=0 ${PY} ${REPO_ROOT}/scripts/parse.py page.png -o out/page"
