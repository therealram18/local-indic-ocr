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

**Document parsing for English and 22 Indian languages, printed and handwritten.** A page image
in; reading-ordered Markdown out, with math as LaTeX and tables as HTML or Markdown, plus
per-block JSON.

<div align="center">

<img src="assets/diagram.png" alt="IndicDocParser: page image to layout detection with reading order, then block-level OCR, then Markdown" width="100%">

</div>

IndicDocParser reads a document page and returns its text in reading order. It is a modular,
two-stage parser: **IndicDocLayout** finds the blocks on the page and orders them, and
**IndicBlockOCR** transcribes each block, with math produced in LaTeX format, and tables as HTML or Markdown.

Since the stages are separate, the layout module can be used as is or swapped by a different one, or
run only the block-level text recognizer. The intermediate step is a structured JSON file passed between the two.

---

<h2 id="examples" style="color:#F97316;">Examples</h2>

Detected blocks with their reading order on the left, the transcription on the right.

<div align="center">
  <img src="assets/gallery-1-english-math-ramanujan.png" alt="A page from Ramanujan's notebooks: text and display equations detected in reading order, with the transcription rendering the mathematics as LaTeX" width="100%">
  <p><b>Example #1 — English page with dense mathematics.</b></p>
</div>

<div align="center">
  <img src="assets/gallery-2-telugu-novel.png" alt="A printed Telugu novel page: paragraph blocks and a page number detected and numbered in reading order, with the Telugu transcription beside it" width="100%">
  <p><b>Example #2 — Printed Telugu page.</b></p>
</div>

<div align="center">
  <img src="assets/cand-hindi-maths-g10-6pr6eq-7dcd5d95-p20.png" alt="A handwritten Hindi maths exercise on ruled paper: alternating Equation and Paragraph blocks detected in reading order, with the transcription rendering the algebra as LaTeX" width="100%">
  <p><b>Example #3 — Handwritten Hindi maths.</b></p>
</div>

---

<h2 id="the-two-models" style="color:#F97316;">The two models</h2>

| | IndicDocLayout | IndicBlockOCR |
| --- | --- | --- |
| **Role** | detect blocks + reading order | transcribe each block |
| **Architecture** | PP-DocLayoutV3 (RT-DETR) for layout detection and reading order | Qwen3.5 for text recognition |
| **Parameters** | 33 M | 0.8 B |
| **Precision** | fp32 | bf16 |
| **In this repo** | `weights/layout` (133 MB) | `weights/ocr` (1.7 GB) |

IndicBlockOCR is trained using the **Sarvam-30B tokenizer**, which has a large vocabulary tuned for good performance on Indian scripts. IndicDocLayout is a finetune based on PP-DocLayoutV3/RT-DETR, trained to predict from an education-domain focussed taxonomy of 37-classes.

---

<h2 id="supported-languages" style="color:#F97316;">Supported languages</h2>

**Printed** page recognition is supported across English and the 22 constitutionally recognised Indian languages: Assamese, Bengali, Bodo, Dogri,
Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali,
Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.

**Handwriting** recognition is supported for English and
12 languages: Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Odia,
Punjabi, Assamese and Urdu. We are very excited to take this first step in supporting handwriting recognition across many Indian languages, and are working on improving quality further, and adding support for other languages as well, in next iterations.

---

<h2 id="how-to-use-this-model" style="color:#F97316;">How to Use this Model</h2>

<h3 style="color:#F97316;">Installation</h3>

Download the repo first, then run its installer — it picks the CUDA wheels that match your driver:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('bodhan-ai/indic-doc-parser'))"
cd <the path it printed> && ./install.sh
```

Or by hand, in **two** commands:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # your CUDA line
pip install transformers accelerate pillow numpy huggingface_hub
```

> **The first command is one command on purpose.** `torchvision` pins an exact `torch`, so
> installing them separately lets pip pick a mismatched pair. And plain `pip install torch` takes
> the newest wheel — currently CUDA 13 — which on an older driver initialises to CPU with only a
> warning, giving you a working install that never touches the GPU. Use the index for *your* CUDA:
> `cu121`, `cu124`, `cu128`, `cu130`, or `cpu`.

**Verified configurations**

| CUDA | torch | torchvision | transformers | numpy |
| --- | --- | --- | --- | --- |
| 12.4 | 2.6.0 | 0.21.0 | 5.15.0 | 2.5.2 |
| 12.9 | 2.11.0 | 0.26.0 | 5.14.1 | 2.3.5 |

Both parse the Example #1 page identically. The span between them is wide, so anything in range
should work — these are the two we ran end to end.

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


```python
layout = parser.detect("page.png")          # {image, width, height, blocks}
```

It also needs neither a GPU nor `torchvision`. On CPU the layout stage takes about as long as on
GPU — 7.8 s versus 8.5 s for the Example #1 page — so `IndicDocParser.from_pretrained(repo,
device="cpu")` is a real option for layout work. The recognizer is a different matter: it runs on
CPU but is far too slow to be practical.

<h3 style="color:#F97316;">OCR Model</h3>


```python
from transformers import AutoModelForImageTextToText, AutoProcessor

proc = AutoProcessor.from_pretrained(repo, subfolder="weights/ocr")
rec = AutoModelForImageTextToText.from_pretrained(repo, subfolder="weights/ocr")
```


```python
from idp_model_ppdoc import PPDocLayoutV3Trainable

model = PPDocLayoutV3Trainable.from_pretrained(f"{repo}/weights/layout")
```

Both are finetunable with standard Hugging Face tooling.

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
| Sarvam-OCR  | 84.3 | 86.5 | 99.6 | 96.3 | 91.0 | 82.2 | 49.8 | 81.0 | 88.3 |
| PaddleOCR-VL-1.6  | 78.7 | 85.1 | 98.7 | 95.8 | 75.1 | 84.1 | 39.0 | 68.8 | 82.9 |

Re-measurement under the shipped configuration is pending.

<h3 style="color:#F97316;">Printed Recognition Performance</h3>

Needs reevaluation.

<h3 style="color:#F97316;">Handwriting Recognition Performance</h3>

Needs reevaluation.

---

<h2 id="limitations" style="color:#F97316;">Limitations</h2>

- We are currently working on better support for **complex multi-column reading order**. Additionally, we are working on better support for **Handwriting** across children versus adult handwriting, and also extending support to other Indic languages.

---

<h2 id="troubleshooting" style="color:#F97316;">Troubleshooting</h2>

Nearly every install problem is the torch/torchvision/CUDA triangle.

**`ImportError` after `ImportError`, or complaints that torchvision is too old.** `torchvision`
pins an exact `torch`, so if the two were installed separately pip may have paired them wrongly.
Reinstall both in one command, from one index:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**`CUDA initialization: The NVIDIA driver on your system is too old`, then everything runs on CPU.**
Plain `pip install torch` fetched a wheel built for a newer CUDA than your driver. Check with
`nvidia-smi`, then reinstall from the matching index. Verify with:

```python
import torch; print(torch.__version__, torch.cuda.is_available())
```

**`Qwen3VLVideoProcessor requires the Torchvision library`.** The recognizer needs `torchvision`;
the layout stage does not. Install it *together with* torch, from the same index.

**`RTDetrHungarianMatcher requires the scipy library`.** You are on an old revision of this repo.
Update — inference no longer builds the matcher.

---

<h2 id="hardware" style="color:#F97316;">Hardware</h2>

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
