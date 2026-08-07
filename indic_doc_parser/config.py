"""Pipeline configuration. NOTE: ocr_lib reads several of these knobs at IMPORT time
(OCR_MIN_PX_SIDE, DEDUP_MODE, OCR_PROMPT, ...), so apply() must run BEFORE ocr_lib is imported.
pipeline.py does this. Override any of them via environment variables before importing the package.
"""
import os
import sys

# The OCR model JIT-compiles a GDN kernel on first inference and needs `ninja` on PATH.
# The venv's bin/ (where ninja lives) isn't on PATH when python is invoked by full path — add it.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

# vLLM stability flags for this checkpoint (set before the engine imports).
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# The 82.9 "clean" recipe (English olmOCR-Bench). Env vars already set take precedence.
DEFAULTS = {
    "OCR_MIN_PX_SIDE": "256",      # crop-upscale floor (largest single win)
    "DEDUP_NEST": "1",
    "DEDUP_MODE": "text_only",     # drop inline eqn boxes (the math prompt covers them)
    "OCR_PROMPT": ("Transcribe the text in this image. Write any mathematical expressions in "
                   "LaTeX, using $...$ for inline math and $$...$$ for display equations."),
}

# Layout inference + engine defaults (not read by ocr_lib at import).
LAYOUT_CONF = 0.5
LAYOUT_IMG_SIZE = 1024
GPU_MEMORY_UTILIZATION = 0.80


def apply():
    for k, v in DEFAULTS.items():
        os.environ.setdefault(k, v)
