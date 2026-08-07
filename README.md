---
language:
  - en
  - as
  - bn
  - brx
  - doi
  - gom
  - gu
  - hi
  - kn
  - ks
  - mai
  - ml
  - mni
  - mr
  - ne
  - or
  - pa
  - sa
  - sat
  - sd
  - ta
  - te
  - ur
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
inference: false
---

# IndicDocParser — Private Preview

**A block-level document parser for English and Indian-language pages.** It reads a page image
and returns the page as reading-ordered markdown plus a per-block JSON — layout, reading order,
and text recognition in one pass.

IndicDocParser is a *modular* parser: a layout+reading-order model finds and orders the blocks on
a page, and a separate recognizer transcribes each block. Because recognition is per-block, the
recognizer can be swapped or upgraded without retraining the layout front-end.

> ### ⚠️ Confidential — Private Preview
>
> This repository and its weights are shared under a confidentiality agreement for **evaluation
> purposes only**. Do not redistribute the weights, publish outputs attributed to these models, or
> deploy them in production without written authorisation from Bodhan AI.

---

## Model summary

| | |
|---|---|
| **Name** | IndicDocParser-Private-Preview |
| **Release** | Private Preview, August 2026 |
| **Task** | Document parsing — layout detection + reading order + block OCR (image → markdown + JSON) |
| **Stage 1 — layout** | `PPDocLayoutV3Trainable` (RT-DETR / PP-DocLayoutV3 subclass with an integrated reading-order head), 33 M params, 133 MB |
| **Stage 2 — OCR** | `Qwen3_5ForConditionalGeneration` block recognizer (*Bodhan Scribe*), 0.8 B params, 1.7 GB |
| **Runtime** | one process, one GPU: layout (torch) + OCR (vLLM) in a single environment |
| **Precision** | bfloat16 |
| **Output** | reading-ordered markdown (LaTeX math, markdown tables) + per-block JSON |

---

## Capabilities

- **Layout + reading order in one model.** Blocks (paragraphs, headings, lists, tables, equations,
  captions, footnotes) are detected *and* ordered by a single RT-DETR-based model with a precedence
  head — not a detector followed by a geometric heuristic.
- **Per-block recognition.** Each block is transcribed with a prompt matched to its type: prose and
  headings as text (with inline `$…$` / display `$$…$$` LaTeX for any math), standalone equations as
  LaTeX, tables as GitHub-flavored markdown.
- **Reading-ordered reconstruction.** Blocks are stitched back in reading order and de-hyphenated
  (words split across a line break are rejoined), so the markdown reads as continuous text.
- **Running furniture is dropped.** Page numbers, folios, headers/footers, and figures are removed
  from the transcription so the output is content, not decoration.
- **Indic-first.** The recognizer is trained for Indian-language scripts and English, including
  math-in-context.

---

## Example

A page from an Assamese legal text (unedited pipeline output).

**`page.md`** (excerpt):

```markdown
অধ্যায়—৬

ন্যায়নির্ণয়

২৮। ধাৰা ২৫ৰ অধীনত অভিগৃহীত কোনো সামগ্ৰী বা প্ৰচলিত মুদ্ৰা বা প্ৰতিভূতি বাজেয়াপ্তকৰণলৈ
দায়ী হ'ব যদিহে এনে সামগ্ৰী বা প্ৰচলিত মুদ্ৰা বা প্ৰতিভূতিক ধাৰা ২৯ৰ অধীনত …
```

**`page.json`** (one block):

```json
{
  "order": 3,
  "type": "SectionHeader",
  "label": "Section-title",
  "bbox_xyxy": [844.9, 361.1, 997.9, 406.3],
  "conf": 0.784,
  "text": "অধ্যায়—৬"
}
```

---

## Environment setup

Both models load in **one environment**. The one script in this repo builds it, matching CUDA
wheels to your driver:

```bash
bash envs/setup_inference_env.sh          # creates ./.venv_idp (venv, not conda)
VENV_DIR=~/idp bash envs/setup_inference_env.sh --recreate
```

Or install by hand into a GPU env you already have:

```bash
CUDA_TAG=cu129                            # the CUDA line your driver supports
pip install torch==2.11.0 torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install vllm --extra-index-url "https://wheels.vllm.ai/nightly/${CUDA_TAG}" \
                 --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install transformers==5.7.0 accelerate huggingface_hub hf_xet ninja pillow numpy
```

Notes:
- **`transformers` ≥ 5.7** — the layout model uses the `PPDocLayoutV3` classes added there.
- **vLLM** must support the OCR model's Qwen3.5 GDN kernels; the reference env used a `cu129` build
  from vLLM's per-CUDA index. **`ninja` must be on `PATH`** — the OCR model JIT-compiles a kernel on
  first inference (`scripts/parse.py` puts the venv's `bin/` on `PATH` for you).
- The setup script's final gate catches the two silent failures — a CUDA/driver wheel mismatch and a
  missing `ninja` — before a model load does.

---

## Quickstart

Download the repo (code + both model weights, ~1.85 GB), then run the script:

```bash
huggingface-cli download oikantik-bodhan/indic-doc-parser --local-dir indic-doc-parser
cd indic-doc-parser

# one page
CUDA_VISIBLE_DEVICES=0 python scripts/parse.py page.png -o out/page      # -> out/page.md + out/page.json

# a directory of pages
CUDA_VISIBLE_DEVICES=0 python scripts/parse.py pages/ -o out/

# inspect the prompts / block taxonomy / output schema
python scripts/parse.py --show-contract
```

`scripts/parse.py` finds `weights/{layout,ocr}` next to itself — nothing to configure.

Python API (optional `pip install -e .`):

```python
from indic_doc_parser import IndicDocParser
p = IndicDocParser()                 # loads OCR (vLLM) then layout onto the current GPU
r = p.parse("page.png")
print(r["markdown"])
print(r["blocks"][0])                # {order, type, label, bbox_xyxy, conf, text}
```

---

## Output contract

| File | Contents |
|---|---|
| `<name>.md` | Reconstructed, reading-ordered, de-hyphenated markdown for the page |
| `<name>.json` | `{"image": str, "blocks": [ … ]}` in reading order |

Each block: `order` (int, reading rank) · `type` · `label` · `bbox_xyxy` (`[x0,y0,x1,y1]` px) ·
`conf` (layout confidence) · `text` (LaTeX for math, markdown for tables). The prompts, the kept
block types, and the dropped-furniture labels are the single-source-of-truth in
[`scripts/idp_contract.py`](scripts/idp_contract.py).

---

## Recommended settings

The default recipe is the olmOCR-Bench-English "clean" configuration. Override via env vars **before
import**, or the `scripts/parse.py` flags:

| Knob | Default | Flag | Effect |
|---|---|---|---|
| `OCR_MIN_PX_SIDE` | `256` | `--min-px-side` | crop-upscale floor (largest single win) |
| `DEDUP_MODE` | `text_only` | `--dedup-mode` | drop inline equation boxes covered by the math prompt |
| layout conf | `0.5` | — | block detection threshold |
| layout img size | `1024` | — | layout input resolution (matches training) |
| `gpu_memory_utilization` | `0.80` | `--gpu-memory-utilization` | vLLM KV-cache budget |

Recognition is greedy (temperature 0) and deterministic — the same page yields the same output.

---

## Hardware

| | |
|---|---|
| **Weights** | 1.85 GB (layout 133 MB + OCR 1.7 GB), bfloat16 |
| **Minimum GPU** | 24 GB (vLLM claims 80% for KV cache; the layout model shares the rest) |
| **Recommended** | 40–80 GB (A100 / H100) |
| **Precision** | bfloat16, unquantised |

## Score

olmOCR-Bench (English): **82.9** clean / 84.1 max — beats sherry V3 (82.5) and PaddleOCR-VL-1.6
fresh (78.7) on the same eval. Known weakness: multi-column reading order.

---

## Files in this repository

| Path | Purpose |
|---|---|
| `weights/layout/` | Layout+RO checkpoint + modeling code + `auto_map` (standalone-loadable) |
| `weights/ocr/` | OCR checkpoint — standard format, vLLM-ready |
| `scripts/parse.py` | The pipeline CLI — image(s) → `.md` + `.json` |
| `scripts/idp_contract.py` | Prompts, block taxonomy, output schema — the contract |
| `envs/setup_inference_env.sh` | Builds the venv with CUDA-matched wheels and a verification gate |
| `indic_doc_parser/` | The engine (importable; `pip install -e .` optional) |
| `pyproject.toml`, `requirements.txt` | Packaging metadata |

---

## License and terms

The layout model derives from PP-DocLayoutV3 / RT-DETR; the recognizer derives from the Qwen model
family — the respective upstream licenses pass through to you. In addition, this Private Preview
checkpoint is supplied under the confidentiality terms of your evaluation agreement with Bodhan AI;
the weights may not be redistributed or used to train other models without written authorisation.

---

<sub>IndicDocParser-Private-Preview · Private Preview, August 2026 · © Bodhan AI · Confidential</sub>
