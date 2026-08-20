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
two-stage parser: **IndicDocLayout** detects the blocks on the page and orders them, and
**IndicBlockOCR** transcribes the textual blocks. The two stages communicate through a structured
JSON file, so either stage can be used independently or replaced with another implementation.

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
IDP=$(python -c "from huggingface_hub import snapshot_download as d; print(d('bodhan-ai/indic-doc-parser'))")
cd "$IDP" && ./install.sh
```

It will use `uv` if that is available, and `pip` otherwise. Where running a shell script is not
convenient, [TROUBLESHOOTING.md](TROUBLESHOOTING.md) lists the two commands it runs.

<h3 style="color:#F97316;">Basic inference</h3>

```python
import sys
from huggingface_hub import snapshot_download

repo = snapshot_download("bodhan-ai/indic-doc-parser")
sys.path.insert(0, repo)                    # the code ships in the repo
from indic_doc_parser import IndicDocParser

parser = IndicDocParser.from_pretrained(repo)

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
from indic_doc_parser import IndicDocLayout, IndicBlockOCR

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

<h3 style="color:#F97316;">Table format</h3>

Tables come back as HTML by default. Choose the format when you construct the parser:

```python
parser = IndicDocParser.from_pretrained(repo)                            # HTML (default)
parser = IndicDocParser.from_pretrained(repo, table_format="markdown")   # Markdown
```

---

<h2 id="performance" style="color:#F97316;">Performance</h2>

<h3 style="color:#F97316;">OmniDocBench 1.6 (eng-official)</h3>

| OmniDocBench 1.6 (eng-official) | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | TableTEDS-S↑ | Read OrderEdit↓ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| PaddleOCRVL-1.6 | 96.36 | 0.03 | 98.55 | 93.37 | 96.33 | 0.09 |
| Sarvam Vision (reported) | 93.28 | 0.02 | 96.54 | 85.26 | 92.83 | 0.03 |
| Chandra OCR | 93.11 | 0.04 | 96.93 | 86.07 | 90.34 | 0.09 |
| Gemini 3.5 Flash | 92.94 | 0.04 | 96.20 | 86.57 | 92.17 | 0.14 |
| **IndicOCR (ours)** | **92.76** | **0.04** | **97.53** | **85.10** | **90.58** | **0.11** |
| GPT-5.6-sol | 92.46 | 0.04 | 95.42 | 85.87 | 90.98 | 0.10 |
| Gemini 3.1 Pro | 91.15 | 0.06 | 95.53 | 83.46 | 88.77 | 0.13 |
| Surya OCR 2 (model) | 91.13 | 0.04 | 95.67 | 81.61 | 86.37 | 0.10 |
| Gemma 31B | 86.71 | 0.0915 | 89.48 | 79.79 | 85.19 | 0.1913 |

<h3 style="color:#F97316;">olmOCR-Bench (eng-sarvam)</h3>

| OlmoOCRBench (eng-sarvam) | Overall | arxiv_math | baseline | headers_footers | long_tiny_text | multi_column | old_scans | old_scans_math | table_tests |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Chandra OCR | 86 | 86.7 | 99.8 | 91.5 | 93.7 | 84.7 | 51 | 88.2 | 92.2 |
| Sarvam Vision (reported) | 84.3 | 86.5 | 99.6 | 96.3 | 91 | 82.2 | 49.8 | 81 | 88.3 |
| Gemini 3.5 Flash | 83.2 | 84.6 | 99.7 | 82.9 | 94.1 | 84.7 | 53.4 | 78.6 | 87.3 |
| Gemini 3.1 Pro | 82.6 | 90.5 | 99 | 82.9 | 88.5 | 81.6 | 47 | 84.3 | 87.3 |
| **IndicOCR (ours)** | **82.2** | **83.2** | **99.4** | **92.9** | **89.8** | **76** | **48.3** | **77.7** | **90** |
| Surya OCR 2 (model) | 81.4 | 82.5 | 99.8 | 92.9 | 79.9 | 85.1 | 42.8 | 84.3 | 84.2 |
| Gemma 31B | 80.4 | 79 | 99.4 | 92.9 | 89.8 | 80.5 | 45.8 | 73.8 | 82.2 |
| PaddleOCRVL-1.6 | 78.7 | 85.1 | 98.4 | 96.2 | 75.3 | 83.9 | 39 | 68.3 | 83 |
| GPT | 78 | 79.3 | 93.9 | 95.4 | 87.8 | 77.4 | 43.7 | 64.6 | 82.2 |

<h3 style="color:#F97316;">IndicOCR-PR — WER by language (lower is better)</h3>

| Language | IndicOCR (ours) | Gemini 3.1 Pro | Gemini 3.5 Flash | SuryaOCR |
| --- | :---: | :---: | :---: | :---: |
| **Overall** | **14.61%** | 19.00% | 22.00% | 32.00% |
| `as` | 10.13% | 9.26% | 11.61% | 13.62% |
| `bd` | 13.54% | 9.03% | 16.36% | 44.44% |
| `bn` | 8.61% | 7.47% | 8.91% | 18.91% |
| `do` | 18.25% | 16.34% | 22.19% | 39.53% |
| `en` | 3.65% | 2.33% | 2.21% | 6.24% |
| `gu` | 8.34% | 7.20% | 8.89% | 20.41% |
| `hi` | 3.94% | 3.69% | 4.48% | 9.71% |
| `kk` | 6.34% | 6.53% | 9.90% | 9.45% |
| `kn` | 12.53% | 10.16% | 13.91% | 24.34% |
| `ks` | 50.13% | 61.95% | 67.90% | 76.55% |
| `ml` | 10.48% | 9.43% | 12.30% | 23.53% |
| `mp` | 17.61% | 99.19% | 99.68% | 99.88% |
| `mr` | 6.13% | 5.48% | 6.19% | 15.72% |
| `mt` | 17.51% | 13.27% | 16.21% | 32.36% |
| `ne` | 8.42% | 6.28% | 8.19% | 12.37% |
| `or` | 25.41% | 15.19% | 26.36% | 35.48% |
| `pa` | 6.96% | 6.53% | 10.31% | 13.69% |
| `sa` | 23.67% | 16.28% | 21.82% | 42.22% |
| `sd` | 13.47% | 13.68% | 16.17% | 19.52% |
| `sn` | 26.55% | 99.76% | 99.26% | 99.89% |
| `ta` | 8.91% | 5.59% | 7.03% | 20.09% |
| `te` | 18.12% | 14.55% | 18.23% | 36.94% |
| `ur` | 13.93% | 11.98% | 12.36% | 23.64% |

<h3 style="color:#F97316;">IndicOCR-HW — WER by language (lower is better)</h3>

| Language | Gemini 3.1 Pro | Gemini 3.5 Flash | IndicOCR (ours) | SuryaOCR |
| --- | :---: | :---: | :---: | :---: |
| **Overall** | 28.00% | 32.50% | **34.38%** | 77.00% |
| `as` | 28.37% | 32.95% | 34.82% | 82.22% |
| `bn` | 25.15% | 29.53% | 29.64% | 90.04% |
| `en` | 15.62% | 18.97% | 17.87% | 27.27% |
| `gu` | 40.04% | 41.33% | 43.72% | 88.51% |
| `hi` | 16.87% | 20.33% | 22.47% | 57.32% |
| `kn` | 26.19% | 33.67% | 32.18% | 86.78% |
| `ml` | 36.07% | 41.45% | 39.95% | 88.18% |
| `mr` | 20.96% | 26.84% | 30.11% | 71.24% |
| `or` | 33.32% | 38.53% | 37.06% | 80.10% |
| `pa` | 29.90% | 35.11% | 30.80% | 84.30% |
| `ta` | 19.52% | 22.88% | 24.97% | 83.21% |
| `te` | 28.04% | 33.56% | 45.70% | 85.42% |
| `ur` | 45.61% | 47.62% | 53.68% | 77.39% |

---

<h2 id="limitations" style="color:#F97316;">Limitations</h2>

Reading order remains a challenge for **complex, multi-column layouts**. Handwriting recognition
is also still being improved, particularly across different writing styles and writing
characteristics.

We are also extending handwriting support to additional Indic languages.


---

<h2 id="hardware" style="color:#F97316;">Hardware</h2>

Latency and throughput numbers to follow.

---

<h2 id="license" style="color:#F97316;">License</h2>

This model and repository are released under the **Apache 2.0 License**.

The release incorporates components distributed under Apache 2.0, including PP-DocLayoutV3,
Qwen3.5, and the Sarvam-30B tokenizer. See the repository license and the corresponding upstream
licenses for the applicable terms and attribution requirements.
---

<h2 id="citation" style="color:#F97316;">Citation</h2>

```bibtex
@misc{indicdocparser2026,
  title  = {IndicDocParser: Multilingual Document Parsing for English and 22 Indian Languages},
  author = {Bodhan.AI},
  year   = {2026},
  url    = {https://huggingface.co/bodhan-ai/indic-doc-parser}
}
```
