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
license: other
pipeline_tag: image-to-text
tags:
  - ocr
  - document-parsing
  - layout-analysis
  - reading-order
  - indic
  - vision-language-model
metrics:
  - cer
inference: false
---

<h1 id="indicdocparser" style="color:#C1440E;">IndicDocParser</h1>

<div align="center">

[![Pipeline](https://img.shields.io/badge/Pipeline-Layout%20%2B%20OCR-C1440E?style=flat)](#model-architecture)
[![Layout](https://img.shields.io/badge/Layout-RT--DETR%20~33M-C1440E?style=flat)](#model-architecture)
[![Recognizer](https://img.shields.io/badge/Recognizer-Qwen3.5%200.8B-C1440E?style=flat)](#model-architecture)
[![Languages](https://img.shields.io/badge/Languages-23-C1440E?style=flat)](#supported-languages)
[![License](https://img.shields.io/badge/License-Private%20Preview-C1440E?style=flat)](#license--terms-of-use)

</div>

**Block-level document parsing for English and 22 Indian languages — page image in, reading-ordered Markdown and per-block JSON out.**

<!-- TODO(doubt): architecture / pipeline diagram, e.g. assets/pipeline-diagram.png -->

**Quick links** — [Blog](#) · [Demo](#) · [Try it out](#) · [GitHub](https://github.com/AshwinSankar17/bodhan_gen_ai_tools)

**IndicDocParser** reads a document page and returns its text in reading order. It is a **modular, two-stage** parser: a **layout** model finds the blocks on the page (paragraphs, headings, tables, equations, figures) and orders them, and a **recognizer** transcribes each block. Math comes back as LaTeX and tables as Markdown.

Because the stages are separate, you can take the layout on its own, swap in your own layout, or run only the recognizer — the layout intermediate is a plain JSON file passed between the two steps.

> ### ⚠️ Confidential — Private Preview
> These weights are shared for evaluation only. Please do not redistribute them or publish outputs attributed to the model without written permission from Bodhan AI.

---

<h2 id="why-choose-indicdocparser" style="color:#C1440E;">Why Choose IndicDocParser?</h2>

- 🧩 **Modular two stages.** Layout and OCR are separate steps with a plain-JSON handoff, so you can inspect or replace either half.
- 🔢 **Layout with reading order.** A single model detects blocks *and* orders them, so the output reads top-to-bottom, column-by-column — not a bag of boxes.
- 🧮 **Math as LaTeX, tables as Markdown.** Equations are transcribed to LaTeX (`$…$` / `$$…$$`) and tables to GitHub-flavored Markdown, inline with the surrounding prose.
- 🇮🇳 **Indic-first.** Trained for Indian scripts and English, across printed and handwritten pages.
- 🧹 **Furniture dropped.** Page numbers, running headers and footers, and figures are left out of the transcribed text.

---

<h2 id="layout--reading-order" style="color:#C1440E;">Layout &amp; Reading Order</h2>

The layout stage on real pages: coloured boxes are the detected blocks (colour = type), and the numbered badge on each block is its **reading-order** position. Click any page to open it full size.

<table>
<tr>
<td align="center"><a href="assets/layout-hindi.png"><img src="assets/layout-hindi-thumb.png" width="150"/></a><br/><sub>Hindi · magazine</sub></td>
<td align="center"><a href="assets/layout-english.png"><img src="assets/layout-english-thumb.png" width="150"/></a><br/><sub>English · newspaper</sub></td>
<td align="center"><a href="assets/layout-telugu.png"><img src="assets/layout-telugu-thumb.png" width="150"/></a><br/><sub>Telugu · textbook</sub></td>
<td align="center"><a href="assets/layout-bengali.png"><img src="assets/layout-bengali-thumb.png" width="150"/></a><br/><sub>Bengali · notice</sub></td>
<td align="center"><a href="assets/layout-santali.png"><img src="assets/layout-santali-thumb.png" width="150"/></a><br/><sub>Santali · Ol Chiki</sub></td>
</tr>
</table>

---

<h2 id="supported-languages" style="color:#C1440E;">Supported Languages</h2>

The model covers **English and the 22 constitutionally recognised Indian languages**, across printed and handwritten pages.

| Group | Languages |
| --- | --- |
| **English** | English |
| **22 constitutionally recognised languages** | Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu |

---

<h2 id="model-architecture" style="color:#C1440E;">Model Architecture</h2>

IndicDocParser is a pipeline of two models that run in one process on one GPU.

| | Layout | Recognizer |
| --- | --- | --- |
| **Role** | Detect blocks + reading order | Transcribe each block |
| **Architecture** | RT-DETR / PP-DocLayoutV3 with an integrated reading-order head | `Qwen3.5` vision-language model (*Bodhan Scribe*) |
| **Base model** | PP-DocLayoutV3 (document-pretrained) | Qwen3.5-0.8B VLM |
| **Parameters** | ~33M | ~0.8B |
| **Precision** | fp32 | bf16 |
| **Checkpoint size** | 133 MB | 1.7 GB |
| **Runtime** | torch | vLLM |

The recognizer uses the **Sarvam tokenizer**, a large vocabulary tuned for Indian scripts; that vocabulary is the main reason it is a little larger than English-only recognizers of a similar shape.

---

<h2 id="results-at-a-glance" style="color:#C1440E;">Results at a Glance</h2>

<!-- TODO(doubt): performance numbers to be added later (per instruction). -->

Document-parsing quality is measured on **olmOCR-Bench**. Full tables are in [Performance](#performance).

---

<h2 id="how-to-use-this-model" style="color:#C1440E;">How to Use this Model</h2>

<h3 style="color:#C1440E;">Get the model</h3>

Download the repository (code and both model weights) from the Hub. The weights are private, so log in first.

```bash
huggingface-cli login
huggingface-cli download oikantik-bodhan/indic-doc-parser --local-dir indic-doc-parser
cd indic-doc-parser
```

<h3 style="color:#C1440E;">Set up the environment</h3>

One GPU, one environment. The helper script matches CUDA wheels to your driver (vLLM's default PyPI wheel targets a newer CUDA than many drivers provide), builds a venv, and checks the GPU before you rely on it.

```bash
bash envs/setup_inference_env.sh          # builds ./.venv_idp
```

> Verified stack: vLLM 0.26.1, PyTorch 2.13.0 (cu129), Transformers 5.14.1. Override the CUDA line with `CUDA_TAG=cu126 bash envs/setup_inference_env.sh` if your driver needs it.

<h3 style="color:#C1440E;">Parse a page</h3>

```bash
.venv_idp/bin/python -m indic_doc_parser.cli parse page.png -o out
#   writes out/page.md and out/page.json
```

`page.png` can be a single image or a folder of images.

<h3 style="color:#C1440E;">The two stages</h3>

Run layout and OCR separately; the layout file is the only thing passed between them.

```bash
# stage 1 — layout + reading order (no recognizer loaded)
.venv_idp/bin/python -m indic_doc_parser.cli layout page.png -o out
#   writes out/page.layout.json

# stage 2 — OCR each block in that layout
.venv_idp/bin/python -m indic_doc_parser.cli ocr out/page.layout.json -o out
#   writes out/page.md and out/page.json
```

To use a layout produced elsewhere, pass it directly: `... ocr your_layout.json --image page.png -o out`.

<h3 style="color:#C1440E;">From Python</h3>

Run from the repository directory (or add it to `PYTHONPATH`).

```python
from indic_doc_parser import IndicDocParser, LayoutParser, BlockOCR

# combined
result = IndicDocParser().parse("page.png")
print(result["markdown"])

# or the two stages — swap either half
layout = LayoutParser().detect("page.png")     # stage 1
result = BlockOCR().run("page.png", layout)    # stage 2, layout can be your own
```

---

<h2 id="inputs" style="color:#C1440E;">Input(s)</h2>

**Input Type(s):** Image
**Input Format(s):** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`
**Input Parameters:** Two-dimensional (2D) document page image
**Other Properties:** One page per image; a folder of images is processed as a batch.

---

<h2 id="output" style="color:#C1440E;">Output</h2>

**Output Type(s):** Text (Markdown) and structured JSON
**Output Format(s):** `.md`, `.json`
**Output Parameters:** One-dimensional (1D) text; list of blocks

Two files, named after the input image:

- **`page.md`** — the page as Markdown, blocks joined in reading order, hyphenated line-breaks rejoined. Math is LaTeX; tables are Markdown.
- **`page.json`** — one entry per block, in reading order:

| Field | Meaning |
| --- | --- |
| `order` | position in reading order, starting at 0 |
| `type` | block type: `Text`, `SectionHeader`, `Title`, `ListItem`, `Table`, `Equation`, `Caption`, `Footnote` |
| `label` | the raw layout label before it is simplified to `type` |
| `bbox_xyxy` | box on the page in pixels, `[x0, y0, x1, y1]` |
| `conf` | layout detection confidence, 0 to 1 |
| `text` | the transcription (LaTeX for math, Markdown for tables) |

The **layout stage** additionally writes `page.layout.json` — the same shape without `text`, plus `image_path`, `width`, and `height` — which is the intermediate handed to the OCR stage.

---

<h2 id="software-integration" style="color:#C1440E;">Software Integration</h2>

**Runtime Engine:** vLLM (recognizer) + PyTorch / Transformers (layout)

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Ampere
- NVIDIA Hopper
- NVIDIA Lovelace

**Supported Operating System(s):** Linux

**Minimum GPU:** 24 GB. **Recommended:** 40–80 GB (A100 / H100). bf16, unquantised.

---

<h2 id="performance" style="color:#C1440E;">Performance</h2>

<!-- TODO(doubt): fill from the blessed eval runs; left empty for now per instruction. -->

<h3 style="color:#C1440E;">olmOCR-Bench (English)</h3>

| System | ArXiv | Base | Headers/Footers | Tiny text | Multi-col | Old scans | Old-scan math | Tables | Overall |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IndicDocParser | | | | | | | | | |

<h3 style="color:#C1440E;">IndicDLP (layout, mAP)</h3>

| Metric | IndicDocParser |
| --- | :---: |
| mAP@50 | |
| mAP@50-95 | |
| Reading-order Kendall τ | |

---

<h2 id="limitations" style="color:#C1440E;">Limitations</h2>

- **Complex multi-column reading order** is the main weakness; dense newspaper-style layouts can be mis-ordered.
- **One page per image.** PDF handling is up to the caller (render pages to images first).
- **Handwriting** varies by script and quality; very degraded scans lose accuracy.
- **Furniture is dropped by design** — if you need headers, footers, or page numbers in the output, read them from the layout JSON before reconstruction.

---

<h2 id="license--terms-of-use" style="color:#C1440E;">License / Terms of Use</h2>

<!-- TODO(doubt): confirm final license posture. -->

The layout model derives from PP-DocLayoutV3 and the recognizer from the Qwen model family; the respective upstream licenses apply. These Private-Preview weights are additionally covered by your evaluation agreement with Bodhan AI: access is limited to named evaluators, and the weights may not be redistributed or used to train other models without written permission.

---

<h2 id="use-case" style="color:#C1440E;">Use Case</h2>

Converting document page images — printed or handwritten, English or Indian-language — into reading-ordered Markdown and per-block JSON, with math as LaTeX and tables as Markdown.

---

<h2 id="deployment-geography" style="color:#C1440E;">Deployment Geography</h2>

Global

---

<h2 id="citation" style="color:#C1440E;">Citation</h2>

```bibtex
@misc{indicdocparser2026,
  title  = {IndicDocParser: Block-level Document Parsing for Indian Languages},
  author = {Bodhan AI},
  year   = {2026},
  url    = {https://huggingface.co/oikantik-bodhan/indic-doc-parser}
}
```

---

<h2 id="ethical-considerations" style="color:#C1440E;">Ethical Considerations</h2>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Developers should work with their team to ensure this model meets requirements for the relevant industry and use case, and addresses unforeseen product misuse.
