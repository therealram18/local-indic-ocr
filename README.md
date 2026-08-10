---
license: apache-2.0
language: [en, as, bn, brx, doi, gu, hi, kn, ks, kok, mai, ml, mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur]
pipeline_tag: image-to-text
tags:
  - ocr
  - document-parsing
  - layout-analysis
  - reading-order
  - indic
  - vision-language-model
  - qwen
  - rt-detr
metrics: [wer]
---

<h1 id="indicdocparser" style="color:#F97316;">IndicDocParser</h1>

<div align="center">

[![Pipeline](https://img.shields.io/badge/Pipeline-Layout%20%2B%20OCR-F97316?style=flat)](#the-two-models)
[![Layout](https://img.shields.io/badge/IndicDocLayout-33M-F97316?style=flat)](#the-two-models)
[![Recognizer](https://img.shields.io/badge/IndicBlockOCR-0.8B-F97316?style=flat)](#the-two-models)
[![Languages](https://img.shields.io/badge/Languages-23-F97316?style=flat)](#supported-languages)
[![License](https://img.shields.io/badge/License-Apache--2.0-F97316?style=flat)](#license)

</div>

**Block-level document parsing for English and 22 Indian languages. Page image in,
reading-ordered Markdown and per-block JSON out.**

IndicDocParser reads a document page and returns its text in reading order. It is a modular,
two-stage parser: **IndicDocLayout** finds the blocks on the page and orders them, and
**IndicBlockOCR** transcribes each block. Math comes back as LaTeX, tables as HTML.

Because the stages are separate, you can take the layout on its own, swap in your own layout, or
run only the recognizer. The intermediate is a plain JSON file passed between the two.

---

<h2 id="the-two-models" style="color:#F97316;">The two models</h2>

| | IndicDocLayout | IndicBlockOCR |
| --- | --- | --- |
| **Role** | detect blocks + reading order | transcribe each block |
| **Architecture** | PP-DocLayoutV3 (RT-DETR) for layout detection with an integrated reading-order head | Qwen3.5 vision-language model |
| **Parameters** | ~33 M | ~0.8 B |
| **Precision** | fp32 | bf16 |
| **In this repo** | `weights/layout` (133 MB) | `weights/ocr` (1.7 GB) |

IndicBlockOCR uses the **Sarvam-30B tokenizer**, a large vocabulary tuned for Indian scripts —
the main reason it is a little larger than English-only recognizers of similar shape.

IndicDocLayout is a finetune, not stock PP-DocLayoutV3: it carries a 37-class education-domain
taxonomy rather than the upstream document vocabulary, plus a locality-weighted reading-order
loss. Only **0.057%** of its parameters are taxonomy-specific (three tensors), so retargeting it
to your own label set keeps essentially the whole pretrained detector.

---

<h2 id="supported-languages" style="color:#F97316;">Supported languages</h2>

English and the 22 constitutionally recognised Indian languages: Assamese, Bengali, Bodo, Dogri,
Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali,
Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.

**Printed** pages are supported across all of these. **Handwriting** is supported for English and
12 languages: Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Odia,
Punjabi, Assamese and Urdu.

---

<h2 id="how-to-use-this-model" style="color:#F97316;">How to Use this Model</h2>

<h3 style="color:#F97316;">Installation</h3>

```bash
pip install torch transformers pillow huggingface_hub
```

The model code ships inside this repository, so there is nothing else to install — no vLLM, no
other toolkit, and no `trust_remote_code`.

<h3 style="color:#F97316;">Basic inference</h3>

```python
import sys
from huggingface_hub import snapshot_download

repo = snapshot_download("bodhan-ai/indic-doc-parser")
sys.path.insert(0, repo)                    # the code ships in the repo
from indic_doc_parser import IndicDocParser

parser = IndicDocParser.from_pretrained(repo)
print(parser("page.png"))                   # markdown
```

`sys.path.insert` is needed because `snapshot_download` returns a cache directory, which is not
importable on its own.

For the per-block JSON as well as the markdown:

```python
page = parser.parse("page.png")
page["markdown"]                            # reading-ordered Markdown
page["blocks"]                              # [{order, label, type, bbox_xyxy, conf, text}, ...]
```

<h3 style="color:#F97316;">Layout only</h3>

Stage 1 loads torch and nothing else — no recognizer weights — so it runs on a much smaller GPU,
and its output is a plain dict you can inspect or correct before stage 2 sees it.

```python
layout = parser.detect("page.png")          # {image, width, height, blocks}
```

<h3 style="color:#F97316;">Using either model on its own</h3>

The recognizer is a **stock Qwen3.5 checkpoint** — no custom code, no flag:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

proc = AutoProcessor.from_pretrained(repo, subfolder="weights/ocr")
rec = AutoModelForImageTextToText.from_pretrained(repo, subfolder="weights/ocr")
```

The detector, imported from the snapshot:

```python
from idp_model_ppdoc import PPDocLayoutV3Trainable

model = PPDocLayoutV3Trainable.from_pretrained(f"{repo}/weights/layout")
```

Both are finetunable with standard Hugging Face tooling.

> **If you finetune the recognizer, keep the prompt contract.** Accuracy depends on the per-block
> prompts and on crops being upscaled to a 256 px² floor. Change either and quality drops for
> reasons that are not obvious from the output. The prompts are in `idp_contract.py`.

---

<h2 id="output" style="color:#F97316;">Output</h2>

| field | meaning |
| --- | --- |
| `order` | reading-order rank, 0-based and gap-free |
| `label` | the raw IndicDocLayout class (37-class taxonomy) |
| `type` | coarse pipeline category — `Text`, `Table`, `Equation`, `Title`, … |
| `bbox_xyxy` | pixel box `[x0, y0, x1, y1]` |
| `conf` | detection confidence |
| `text` | transcription; `""` for blocks not sent to the recognizer |

**Nothing is silently dropped.** Figures, running headers and footers are not transcribed, but
they remain in the JSON with `text: ""`, so you can see what was detected and where. Page numbers
and other margin text *are* transcribed.

**Tables are HTML by default.** `colspan`, `rowspan` and in-cell line breaks have no
GitHub-flavored-Markdown spelling, so a merged-cell table rendered as GFM silently loses its
structure. Pass `table_format="markdown"` to `from_pretrained` if your tables are flat.

---

<h2 id="performance" style="color:#F97316;">Performance</h2>

<h3 style="color:#F97316;">olmOCR-Bench (English)</h3>

| System | Overall | ArXiv math | Baseline | Headers/Footers | Long/tiny | Multi-col | Old scans | Old-scan math | Tables |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicDocParser** | **82.9** | 84.5 | 99.2 | 98.3 | 90.3 | 73.5 | 47.3 | 80.8 | 89.5 |
| Sarvam-OCR *(reported)* | 84.3 | 86.5 | 99.6 | 96.3 | 91.0 | 82.2 | 49.8 | 81.0 | 88.3 |
| PaddleOCR-VL-1.6 *(reproduced)* | 78.7 | 85.1 | 98.7 | 95.8 | 75.1 | 84.1 | 39.0 | 68.8 | 82.9 |

> **Read the provenance before quoting these.** The 82.9 run used `dedup_mode="text_only"` with
> Markdown tables, on vLLM with torch 2.13. The shipped defaults are `dedup_mode="both"` with HTML
> tables, and this quickstart path runs on `transformers` rather than vLLM. Re-measurement under
> the shipped configuration is pending.

<h3 style="color:#F97316;">IndicDLP (layout)</h3>

Per-source scores for IndicDocLayout are in `weights/layout/test_metrics.json`: mAP@50 0.58–0.73
and reading-order Kendall τ 0.91–0.98 across seven sources — printed textbooks, magazines,
national archives, and four handwriting sets.

---

<h2 id="limitations" style="color:#F97316;">Limitations</h2>

- **Complex multi-column reading order** is the main weakness. On a two-column worked example the
  two derivations can interleave: every transcription is correct, but their order is not.
- **One page per image.** Render PDF pages to images first.
- **Handwriting** varies by script and quality, and covers the 12 languages listed above.
- **This path has no continuous batching.** It runs the recognizer on `transformers`, roughly 20×
  slower per block than a vLLM deployment. Fine for a page; use vLLM for a corpus.
- **A tokenizer warning you should ignore.** `transformers` may suggest `fix_mistral_regex=True`.
  Do not set it — the default is what the model was trained and evaluated with.

---

<h2 id="hardware" style="color:#F97316;">Hardware</h2>

Linux, one GPU. bf16, unquantised. Ampere, Hopper or Lovelace. The layout stage alone runs
comfortably in a few GB; the full pipeline wants 24 GB or more.

---

<h2 id="license" style="color:#F97316;">License</h2>

**Apache 2.0**, inherited cleanly from everything it is built on: PP-DocLayoutV3 (Apache 2.0),
Qwen3.5 (Apache 2.0), and the Sarvam-30B tokenizer (Apache 2.0). You may use, modify, finetune
and redistribute, with attribution and a copy of the licence.

---

<h2 id="citation" style="color:#F97316;">Citation</h2>

```bibtex
@misc{indicdocparser2026,
  title  = {IndicDocParser: Block-level Document Parsing for Indian Languages},
  author = {Bodhan AI},
  year   = {2026},
  url    = {https://huggingface.co/bodhan-ai/indic-doc-parser}
}
```
