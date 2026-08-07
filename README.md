---
license: other
library_name: indic-doc-parser
pipeline_tag: image-to-text
tags:
  - ocr
  - document-parsing
  - layout-analysis
  - reading-order
  - indic
  - vision-language-model
language:
  - as
  - bn
  - gu
  - hi
  - kn
  - ml
  - mr
  - or
  - pa
  - ta
  - te
  - en
---

# IndicDocParser

Modular, block-level document parser for Indic (and English) pages. Two models, one pipeline:

1. **layout** (`weights/layout`) — an RT-DETR / PPDocLayoutV3 subclass with an integrated
   reading-order head. Detects blocks (paragraphs, headings, tables, equations, figures…) and orders them.
2. **ocr** (`weights/ocr`) — a Qwen-based VLM recognizer (*Bodhan Scribe*) that transcribes each block,
   emitting LaTeX for math and GitHub-flavored markdown for tables.

Output per page: **`<name>.md`** (reconstructed, dehyphenated markdown) and **`<name>.json`**
(one record per block: `order`, `type`, `label`, `bbox_xyxy`, `conf`, `text`).

## Getting started (minimal)

Requires a **CUDA-12.9 GPU env** with the matching `torch 2.11.0+cu129` and
`vllm 0.20.1rc1.dev…+cu129` build (needed for the Qwen3.5 GDN kernels — not on PyPI). If you have
that env, this is the whole flow:

```bash
# 1. download the repo (code + both model weights, ~1.85 GB)
huggingface-cli download oikantik-bodhan/indic-doc-parser --local-dir indic-doc-parser

# 2. install the package into your GPU env
cd indic-doc-parser && pip install -e .

# 3. run  (weights/ are auto-resolved from the downloaded repo)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  indic-doc-parser page.png -o out          # -> out.md + out.json
```

Python API:

```python
from indic_doc_parser import IndicDocParser
p = IndicDocParser()                 # loads OCR (vLLM) then layout onto the current GPU
r = p.parse("page.png")
print(r["markdown"])
print(r["blocks"][0])                # {'order':…, 'type':…, 'bbox_xyxy':…, 'text':…}
```

> Don't have the cu129 env? A Docker image (bakes the torch/vLLM build) is the turnkey path for
> bare boxes — planned; ping the maintainers.

## Config

The winning "clean" recipe is the default. Override via env vars **before import** (the OCR library
reads some at import time): `OCR_MIN_PX_SIDE` (256), `DEDUP_MODE` (`text_only`), `OCR_PROMPT`, …
See `indic_doc_parser/config.py`.

## Score

olmOCR-Bench (English): **82.9** clean / 84.1 max — beats sherry V3 (82.5) and PaddleOCR-VL-1.6 (78.7)
on the same eval.

## Repo layout

```
indic_doc_parser/       # the package (pip install -e .)
├── pipeline.py         # IndicDocParser: layout -> crop -> OCR -> reconstruct -> dehyphenate
├── ocr_lib.py          # OCR / layout-oracle logic (detect/crop/dedup/reconstruct/prompts)
├── config.py · cli.py
└── layout/             # layout+RO model code (modeling_ppdoc, order_loss, labels, infer)
weights/
├── layout/             # layout checkpoint (+ modeling code + auto_map for standalone loading)
└── ocr/                # OCR checkpoint (standard, vLLM-ready)
```
