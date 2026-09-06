---
language:
- en
- as
- bn
- brx
- doi
- gu
- hi
- kn
- ks
- kok
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
---


<div align="center">

[![Pipeline](https://img.shields.io/badge/Pipeline-Layout%20%2B%20OCR-F97316?style=flat)](#the-two-models)
[![Layout](https://img.shields.io/badge/IndicDocLayout-33M-F97316?style=flat)](#the-two-models)
[![Recognizer](https://img.shields.io/badge/IndicBlockOCR-0.8B-F97316?style=flat)](#the-two-models)
[![Languages](https://img.shields.io/badge/Languages-23-F97316?style=flat)](#supported-languages)
[![License](https://img.shields.io/badge/License-Indic%20Open%20Model%20License%20v1.0-C1440E?style=flat#model-badge)](#license-and-citation)

</div>

**Document parsing for English and 22 Indian languages, printed and handwritten.** A page image
in; reading-ordered Markdown out, with math as LaTeX and tables as HTML or Markdown, plus
per-block JSON.

<div align="center">

<img src="assets/diagram.png" alt="IndicOCR: page image to layout detection with reading order, then block-level OCR, then Markdown" width="100%">

</div>

IndicOCR reads a document page and returns its text in reading order. It is a modular,
two-stage parser: **IndicDocLayout** detects the blocks on the page and orders them, and
**IndicBlockOCR** transcribes the textual blocks. The two stages communicate through a structured
JSON file, so either stage can be used independently or replaced with another implementation.

[`ARCHITECTURE.md`](ARCHITECTURE.md) traces one page through the whole call path, names what
each module does, and lists the invariants that break the output silently when violated.

---

<h2 id="examples" style="color:#F97316;">Examples</h2>

Detected blocks with their reading order on the left, the transcription on the right.

<div align="center">
  <img src="assets/gallery-1-english-math-ramanujan.png" alt="A page from Ramanujan's notebooks: text and display equations detected in reading order, with the transcription rendering the mathematics as LaTeX" width="100%">
  <p><b>Example #1. English page with dense mathematics.</b></p>
</div>

<div align="center">
  <img src="assets/gallery-2-telugu-novel.png" alt="A printed Telugu novel page: paragraph blocks and a page number detected and numbered in reading order, with the Telugu transcription beside it" width="100%">
  <p><b>Example #2. Printed Telugu page.</b></p>
</div>

<div align="center">
  <img src="assets/cand-hindi-maths-g10-6pr6eq-7dcd5d95-p20.png" alt="A handwritten Hindi maths exercise on ruled paper: alternating Equation and Paragraph blocks detected in reading order, with the transcription rendering the algebra as LaTeX" width="100%">
  <p><b>Example #3. Handwritten Hindi maths.</b></p>
</div>

---

<h2 id="model-summary" style="color:#F97316;">Model Summary</h2>

| | IndicDocLayout | IndicBlockOCR |
| --- | --- | --- |
| **Role** | Layout detection + reading order | Block-level text recognition |
| **Architecture** | PP-DocLayoutV3 / RT-DETR | Qwen3.5-0.8B |
| **Parameters** | 33 M | 0.8 B |
| **Precision** | fp32 | bf16 |
| **In this repo** | `weights/layout` (133 MB) | `weights/ocr` (1.7 GB) |
| **Output** | Layout JSON | Markdown + block JSON |


IndicBlockOCR uses the **Sarvam-30B tokenizer**, with a vocabulary designed to cover Indian
scripts. IndicDocLayout is a fine-tune of PP-DocLayoutV3/RT-DETR, trained with a 37-class
taxonomy designed for education-domain documents.

IndicDocLayout predicts a labelled bounding box for each detected layout element.
The 37 supported labels are:

> Advertisement, Answer, Author, Chapter-end-section, Chapter-title, Chart, Code, Contact-info, Dateline, Diagram, Equation, Expression, Flag, Folio, Footer, Footnote, Header, Image, Image-caption, Index, Infobox, List, MCQ, Page-number, Paragraph, Placeholder-text, Question, Reference, Section-title, Solved-example, Sub-section-title, Sub-sub-section-title, Table, Table-caption, Table-of-contents, Title, Website-link

---

<h2 id="supported-languages" style="color:#F97316;">Supported languages</h2>

**Printed** page recognition is supported across English and the 22 constitutionally recognised Indian languages: Assamese, Bengali, Bodo, Dogri,
Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali,
Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.

**Handwriting** recognition currently supports English and 12 Indian languages: Hindi, Bengali,
Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, and Urdu.

Handwriting quality is still a work in progress, particularly across different writing styles. We are working on improving recognition and extending support to additional languages.

---

<h2 id="usage" style="color:#F97316;">Usage</h2>

<h3 style="color:#F97316;">Installation</h3>

The repo ships an installer that reads your driver and picks matching CUDA wheels. If you work in a
virtual environment, please activate it first, as the installer installs into whichever
Python is active.

```bash
IDP=$(python -c "from huggingface_hub import snapshot_download as d; print(d('bodhan-ai/indic-ocr'))")
cd "$IDP" && ./install.sh
```

It will use `uv` if that is available, and `pip` otherwise. Where running a shell script is not
convenient, [TROUBLESHOOTING.md](TROUBLESHOOTING.md) lists the two commands it runs.

<h3 style="color:#F97316;">Basic inference</h3>

```python
import sys
from huggingface_hub import snapshot_download

repo = snapshot_download("bodhan-ai/indic-ocr")
sys.path.insert(0, repo)                    # the code ships in the repo
from indic_ocr import IndicOCR

parser = IndicOCR.from_pretrained(repo)

page = parser.parse("page.png")
print(page["markdown"])                     # reading-ordered Markdown
```

`page` also carries the per-block detail, which you can save as follows:

```python
import json

with open("page.json", "w", encoding="utf-8") as f:
    json.dump(page, f, ensure_ascii=False, indent=2)
```

<h3 style="color:#F97316;">Running one stage at a time</h3>

To run the two stages separately:

```python
from indic_ocr import IndicDocLayout, IndicBlockOCR

layout = IndicDocLayout(f"{repo}/weights/layout").detect("page.png")
page = IndicBlockOCR(f"{repo}/weights/ocr").run("page.png", layout)
```

`run()` takes a layout object, a dict, or the path to a layout JSON file.

---

<h2 id="output" style="color:#F97316;">Output</h2>

`parser.parse("page.png")` returns the page metadata and its blocks in reading order:

```json
{
  "image": "sample1.png",
  "width": 800,
  "height": 1273,
  "blocks": [
    {"order": 0, "label": "Header", "type": "PageHeader",
     "bbox_xyxy": [345.6, 51.7, 437.1, 114.7], "conf": 0.6, "text": ""},
    {"order": 1, "label": "Page-number", "type": "PageNumber",
     "bbox_xyxy": [367.8, 78.9, 413.2, 107.4], "conf": 0.747, "text": "229"},
    {"order": 2, "label": "Paragraph", "type": "Text",
     "bbox_xyxy": [77.9, 121.3, 711.8, 199.4], "conf": 0.863,
     "text": "Thus we see that, if we can prove that twice the L.H.S. of (30) ..."}
  ]
}
```

| field | meaning |
| --- | --- |
| `order` | reading-order rank, 0-based and gap-free |
| `label` | the raw IndicDocLayout class (37-class taxonomy) |
| `type` | coarse pipeline category: `Text`, `Table`, `Equation`, `Title`, ... |
| `bbox_xyxy` | pixel box `[x0, y0, x1, y1]` |
| `conf` | detection confidence |
| `text` | transcription; `""` for blocks not sent to the recognizer |

**Note:** Figures, charts, advertisements, running headers, and footers are not sent through the recognizer
by default. They remain in the JSON with `text: ""`, so you can see what was detected and where.
Page numbers and other margin text such as folios are transcribed.

<h3 style="color:#F97316;">Schemas</h3>

Machine-readable JSON Schema for each envelope, in [`schemas/`](schemas):

| file | describes |
| --- | --- |
| `layout_output.schema.json` | The layout file: what **IndicDocLayout** writes and **IndicBlockOCR** reads. Blocks and reading order, before any text is read, so there is **no** `text` key at all. |
| `parse_output.schema.json` | The parsed page shown above. Every block now has `text`; `""` means the block was detected but deliberately not sent to the recognizer. |

A layout from your own detector must use a `label` from the 37-class taxonomy, or declare `type`
explicitly. An unrecognised label is rejected rather than silently read as prose.

<h3 style="color:#F97316;">Table format</h3>

Tables come back as HTML by default. Choose the format when you construct the parser:

```python
parser = IndicOCR.from_pretrained(repo)                          # HTML (default)
parser = IndicOCR.from_pretrained(repo, table_format="markdown") # Markdown
```

---

<h2 id="performance" style="color:#F97316;">Performance</h2>

<h3 style="color:#F97316;">OmniDocBench 1.6 (english subset)</h3>

| OmniDocBench 1.6 (english subset) | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | TableTEDS-S↑ | Read OrderEdit↓ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| PaddleOCRVL-1.6 | 96.36 | 0.03 | 98.55 | 93.37 | 96.33 | 0.09 |
| Chandra OCR 2 | 93.11 | 0.04 | 96.93 | 86.07 | 90.34 | 0.09 |
| **IndicOCR (ours)** | **92.76** | **0.04** | **97.53** | **85.10** | **90.58** | **0.11** |
| GPT-5.6-sol | 92.46 | 0.04 | 95.42 | 85.87 | 90.98 | 0.10 |
| Gemini 3.1 Pro | 91.15 | 0.06 | 95.53 | 83.46 | 88.77 | 0.13 |
| Surya OCR 2 | 91.13 | 0.04 | 95.67 | 81.61 | 86.37 | 0.10 |
| Sarvam Vision | 90.08 | 0.04 | 97.62 | 76.82 | 82.01 | 0.10 |
| Gemma-4-31B | 86.71 | 0.09 | 89.48 | 79.79 | 85.19 | 0.19 |
| Nemotron Parse 2 | 79.12 | 0.159 | 78.94 | 74.32 | 81.09 | 0.29 |

<h3 style="color:#F97316;">olmOCR-Bench (<a href="https://huggingface.co/datasets/sarvamai/olmOCR-Bench-English" style="color:#F97316;">english subset</a>)</h3>

| OlmoOCRBench ([english subset](https://huggingface.co/datasets/sarvamai/olmOCR-Bench-English)) | Overall↑ | arxiv_math↑ | baseline↑ | headers_footers↑ | long_tiny_text↑ | multi_column↑ | old_scans↑ | old_scans_math↑ | table_tests↑ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Chandra OCR 2 | 85.9 | 86.7 | 99.8 | 91.5 | 93.7 | 84.7 | 51 | 88.2 | 92.2 |
| Sarvam Vision | 84.3 | 86.5 | 99.6 | 96.3 | 91 | 82.2 | 49.8 | 81 | 88.3 |
| Gemini 3.1 Pro | 82.6 | 90.5 | 99 | 82.9 | 88.5 | 81.6 | 47 | 84.3 | 87.3 |
| **IndicOCR (ours)** | **82.2** | **83.2** | **99.4** | **92.9** | **89.8** | **76** | **48.3** | **77.7** | **90** |
| Surya OCR 2 | 81.4 | 82.5 | 99.8 | 92.9 | 79.9 | 85.1 | 42.8 | 84.3 | 84.2 |
| Gemma-4-31B | 80.4 | 79 | 99.4 | 92.9 | 89.8 | 80.5 | 45.8 | 73.8 | 82.2 |
| PaddleOCRVL-1.6 | 78.7 | 85.1 | 98.4 | 96.2 | 75.3 | 83.9 | 39 | 68.3 | 83 |
| GPT-5.6-sol | 78 | 79.3 | 93.9 | 95.4 | 87.8 | 77.4 | 43.7 | 64.6 | 82.2 |
| Nemotron Parse 2 | 68.2 | 64 | 96.7 | 90 | 79.6 | 72.8 | 31.9 | 28.6 | 81.8 |

<h3 style="color:#F97316;">IndicOCR-PR: printed accuracy by language (higher is better)</h3>

Word-level accuracy, reported as 100 x (1 - WER).

| Language | Sarvam Vision | **IndicOCR (ours)** | Gemini 3.1 Pro | Surya OCR 2 | Gemma-4-31B | Chandra OCR 2 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 86.6 | **86.2** | 80.4 | 67.9 | 66.3 | 64.2 |
| Assamese | 89.5 | 90.2 | 90.7 | 86.4 | 70.6 | 73.5 |
| Bodo | 91.0 | 86.5 | 91.0 | 55.6 | 68.1 | 46.6 |
| Bengali | 91.6 | 91.4 | 92.5 | 81.1 | 83.9 | 79.2 |
| Dogri | 85.8 | 81.7 | 83.7 | 60.5 | 64.4 | 55.8 |
| English | 96.6 | 97.0 | 97.7 | 93.8 | 97.2 | 91.3 |
| Gujarati | 91.6 | 91.7 | 92.8 | 79.6 | 81.6 | 73.0 |
| Hindi | 95.7 | 96.0 | 96.3 | 90.3 | 93.7 | 89.3 |
| Konkani | 93.6 | 93.7 | 93.5 | 90.5 | 76.9 | 85.5 |
| Kannada | 88.8 | 88.0 | 89.8 | 75.7 | 68.3 | 69.6 |
| Kashmiri | 43.3 | 52.2 | 38.1 | 23.4 | 19.9 | 17.6 |
| Malayalam | 90.6 | 89.9 | 90.6 | 76.5 | 72.0 | 68.3 |
| Manipuri | 81.9 | 83.8 | 0.8 | 0.1 | 0.1 | 0.0 |
| Marathi | 93.9 | 93.5 | 94.5 | 84.3 | 89.1 | 83.1 |
| Maithili | 86.7 | 83.0 | 86.7 | 67.6 | 76.3 | 66.1 |
| Nepali | 92.5 | 91.5 | 93.7 | 87.6 | 87.2 | 82.1 |
| Odia | 77.5 | 75.7 | 84.8 | 64.5 | 38.7 | 62.6 |
| Punjabi | 92.2 | 93.2 | 93.5 | 86.3 | 75.1 | 84.1 |
| Sanskrit | 82.0 | 76.2 | 83.7 | 57.8 | 60.8 | 55.8 |
| Sindhi | 89.2 | 87.1 | 86.3 | 80.5 | 74.5 | 71.4 |
| Santhali | 71.9 | 74.7 | 0.2 | 0.1 | 0.2 | 0.0 |
| Tamil | 94.2 | 91.3 | 94.4 | 79.9 | 83.3 | 79.0 |
| Telugu | 84.3 | 82.3 | 85.5 | 63.1 | 66.6 | 59.6 |
| Urdu | 87.1 | 85.9 | 88.0 | 76.4 | 76.6 | 74.4 |

<h3 style="color:#F97316;">IndicOCR-HW: handwriting accuracy by language (higher is better)</h3>

Word-level accuracy, reported as 100 x (1 - WER).

| Language | Gemini 3.1 Pro | **IndicOCR (ours)** | Sarvam Vision | Gemma-4-31B | Chandra OCR 2 | Surya OCR 2 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 72.0 | **66.7** | 55.4 | 33.9 | 24.7 | 23.0 |
| Assamese | 71.6 | 66.1 | 47.8 | 24.1 | 8.9 | 17.8 |
| Bengali | 74.8 | 71.3 | 58.3 | 35.1 | 6.6 | 10.0 |
| English | 84.4 | 80.7 | 77.7 | 78.5 | 78.2 | 72.7 |
| Gujarati | 60.0 | 55.9 | 39.2 | 23.7 | 11.8 | 11.5 |
| Hindi | 83.1 | 77.6 | 72.3 | 70.7 | 54.6 | 42.7 |
| Kannada | 73.8 | 69.6 | 57.7 | 17.2 | 11.5 | 13.2 |
| Malayalam | 63.9 | 60.5 | 45.6 | 16.0 | 15.7 | 11.8 |
| Marathi | 79.0 | 70.2 | 61.8 | 56.5 | 35.4 | 28.8 |
| Odia | 66.7 | 68.2 | 40.6 | 15.5 | 19.4 | 19.9 |
| Punjabi | 70.1 | 69.4 | 54.8 | 11.7 | 11.5 | 15.7 |
| Tamil | 80.5 | 76.8 | 60.5 | 33.4 | 18.8 | 16.8 |
| Telugu | 72.0 | 53.5 | 59.1 | 32.0 | 20.8 | 14.6 |
| Urdu | 54.4 | 46.4 | 44.4 | 25.6 | 27.6 | 22.6 |

---

<h2 id="limitations" style="color:#F97316;">Limitations</h2>

Reading order remains a challenge for **complex, multi-column layouts**. Handwriting recognition
is also still being improved, particularly across different writing styles and writing
characteristics.

We are also extending handwriting support to additional Indic languages.


---

<h2 id="throughput" style="color:#F97316;">Throughput</h2>

We benchmarked our end-to-end pipeline with vLLM on a single NVIDIA H100 80GB GPU using a diverse
mix of documents (math, tables, handwritten notes, research papers, multi-column, and even complex
newspaper layouts) from the OmniDocBench V1.6 benchmark set (official english subset).

| Concurrency | Pages / s | Output tokens / s | Median page (ms) | p95 page (ms) |
| --- | :---: | :---: | :---: | :---: |
| 32 | 5.13 | 6,594 | 3,295 | 19,243 |
| 64 | 6.26 | 7,935 | 4,813 | 34,196 |
| 128 | 6.34 | 7,736 | 9,337 | 72,970 |
| 256 | 6.27 | 7,700 | 28,188 | 81,229 |

---

<h2 id="license" style="color:#F97316;">License</h2>

Released under [Indic Open Model License v1.0](Bodhan_AI_Open_Model_License.md).

The release incorporates components distributed under Apache 2.0, including PP-DocLayoutV3,
Qwen3.5, and the Sarvam-30B tokenizer. See the repository license and the corresponding upstream
licenses for the applicable terms and attribution requirements.
---

<h2 id="citation" style="color:#F97316;">Citation</h2>

```bibtex
@misc{indicocr2026,
  title  = {IndicOCR: Multilingual Document Parsing for English and 22 Indian Languages},
  author = {Bodhan.AI},
  year   = {2026},
  url    = {https://huggingface.co/bodhan-ai/indic-ocr}
}
```