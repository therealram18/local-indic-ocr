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


<div align="center">

[![Pipeline](https://img.shields.io/badge/Pipeline-Layout%20%2B%20OCR-F97316?style=flat)](#the-two-models)
[![Layout](https://img.shields.io/badge/IndicDocLayout-33M-F97316?style=flat)](#the-two-models)
[![Recognizer](https://img.shields.io/badge/IndicBlockOCR-0.8B-F97316?style=flat)](#the-two-models)
[![Languages](https://img.shields.io/badge/Languages-23-F97316?style=flat)](#supported-languages)
[![License](https://img.shields.io/badge/License-Apache--2.0-F97316?style=flat)](#license)

</div>

**Multilingual document parsing for English and 22 Indian languages. Give it a page image and get
back reading-ordered Markdown plus per-block JSON, with math as LaTeX and tables as HTML or
Markdown.**

<div align="center">

<img src="assets/diagram.png" alt="IndicDocParser: page image to layout detection with reading order, then block-level OCR, then Markdown" width="100%">

</div>

IndicDocParser reads a document page and returns its text in reading order. It is a modular,
two-stage parser: **IndicDocLayout** finds the blocks on the page and orders them, and
**IndicBlockOCR** transcribes each block, with math produced in LaTeX format, and tables as HTML or Markdown.

Since the stages are separate, the layout module can be used as is or swapped by a different one, or
run only the block-level text recognizer. The intermediate step is a structured JSON file passed between the two.

---

<h2 id="the-two-models" style="color:#F97316;">The two models</h2>

| | IndicDocLayout | IndicBlockOCR |
| --- | --- | --- |
| **Role** | detect blocks + reading order | transcribe each block |
| **Architecture** | PP-DocLayoutV3 (RT-DETR) for layout detection with an integrated reading-order head | Qwen3.5 for text recognition |
| **Parameters** | ~33 M | ~0.8 B |
| **Precision** | fp32 | bf16 |
| **In this repo** | `weights/layout` (133 MB) | `weights/ocr` (1.7 GB) |

IndicBlockOCR is trained using the **Sarvam-30B tokenizer**, which has a large vocabulary tuned for good performance on Indian scripts. IndicDocLayout is a finetune based on PP-DocLayoutV3/RT-DETR: it is trained to predict from an education-domain focussed taxonomy of 37-classes.

---

<h2 id="supported-languages" style="color:#F97316;">Supported languages</h2>

**Printed** pages are supported across English and the 22 constitutionally recognised Indian languages: Assamese, Bengali, Bodo, Dogri,
Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali,
Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.

**Handwriting** is supported for English and
12 languages: Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Odia,
Punjabi, Assamese and Urdu. We are very excited to take this first step in supporting handwriting recognition across many Indian languages, and are working on improving quality further, and adding support for other languages as well, in next iterations.

---

<h2 id="how-to-use-this-model" style="color:#F97316;">How to Use this Model</h2>

<h3 style="color:#F97316;">Installation</h3>

```bash
pip install torch transformers pillow huggingface_hub
```


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

Stage 1 never loads the recognizer, so it runs on a much smaller GPU. It returns a plain
dictionary, which you can inspect or correct before stage 2 ever sees it.

```python
layout = parser.detect("page.png")          # {image, width, height, blocks}
```

<h3 style="color:#F97316;">OCR Model</h3>

The recognizer has been trained for strong Indic support on printed and handwritten blocks of text:

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

**Note:** Figures, charts, advertisements and running headers and footers are not sent through the recognizer model by default, but
they remain in the JSON with `text: ""`, so you can see what was detected and where. Page numbers
and other margin text like folio *are* transcribed.

**Tables are HTML by default.** Pass `table_format="markdown"` to `from_pretrained` for Markdown table output.

---

<h2 id="performance" style="color:#F97316;">Performance</h2>

<h3 style="color:#F97316;">olmOCR-Bench (English)</h3>

| System | Overall | ArXiv math | Baseline | Headers/Footers | Long/tiny | Multi-col | Old scans | Old-scan math | Tables |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicDocParser** | **82.9** | 84.5 | 99.2 | 98.3 | 90.3 | 73.5 | 47.3 | 80.8 | 89.5 |
| Sarvam-OCR *(reported)* | 84.3 | 86.5 | 99.6 | 96.3 | 91.0 | 82.2 | 49.8 | 81.0 | 88.3 |
| PaddleOCR-VL-1.6 *(reproduced)* | 78.7 | 85.1 | 98.7 | 95.8 | 75.1 | 84.1 | 39.0 | 68.8 | 82.9 |

Sarvam-OCR's row is their published number; PaddleOCR-VL's is our reproduction. Our own 82.9 was
measured with `dedup_mode="text_only"` and Markdown tables on vLLM, whereas the shipped defaults
are `dedup_mode="both"` with HTML. Re-measurement under the shipped configuration is pending.

<h3 style="color:#F97316;">Printed Recognition Performance</h3>

Needs reevaluation.

<h3 style="color:#F97316;">Handwriting Recognition Performance</h3>

Needs reevaluation.

---

<h2 id="limitations" style="color:#F97316;">Limitations</h2>

- We are currently working on better support for **complex multi-column reading order**. Additionally, we are working on better support for **Handwriting** across children versus adult handwriting, and also extending support to other Indic languages.
- **This path has no continuous batching.** It runs the recognizer on `transformers`, which is far
  slower per block than a vLLM deployment. Fine for a page; use vLLM for a corpus.
- **A tokenizer warning you should ignore.** `transformers` may suggest `fix_mistral_regex=True`.
  Do not set it — the default is what the model was trained and evaluated with.

---

<h2 id="hardware" style="color:#F97316;">Hardware</h2>

Linux, one GPU. bf16, unquantised. Ampere, Hopper or Lovelace. The layout stage alone runs
comfortably in a few GB; the full pipeline wants 24 GB or more.

Latency and throughput numbers to follow.

---

<h2 id="license" style="color:#F97316;">License</h2>

**Apache 2.0**, inherited cleanly from everything it is built on: PP-DocLayoutV3 (Apache 2.0),
Qwen3.5 (Apache 2.0), and the Sarvam-30B tokenizer (Apache 2.0). You may use, modify, finetune
and redistribute, with attribution and a copy of the licence.

---

<h2 id="citation" style="color:#F97316;">Citation</h2>

```bibtex
@misc{indicdocparser2026,
  title  = {IndicDocParser: Multilingual 2-stage Document Parsing for 22 Indian Languages and English},
  author = {Bodhan.AI},
  year   = {2026},
  url    = {https://huggingface.co/bodhan-ai/indic-doc-parser}
}
```
