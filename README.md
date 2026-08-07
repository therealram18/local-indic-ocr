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

**Turn a page image into clean, reading-ordered text.** Give IndicDocParser a scanned or photographed
document page — English or any of 22 Indian languages — and it returns the page as **Markdown** plus a
**per-block JSON**. Math comes back as LaTeX, tables as Markdown.

It works in two steps, which you can run together or on their own:

1. **Layout** — find the blocks on the page (paragraphs, headings, tables, equations, …) and put them in reading order.
2. **OCR** — read the text inside each block.

Because the steps are separate, you can take just the layout, or bring your own layout and run only the OCR.

> ### ⚠️ Confidential — Private Preview
> The weights are shared for evaluation only. Please don't redistribute them or publish outputs
> attributed to the model without written permission from Bodhan AI.

---

## Install

```bash
pip install indic-doc-parser
```

The model weights (~1.85 GB) download automatically the first time you run it. They're private, so
log in to Hugging Face once first:

```bash
huggingface-cli login
```

You need one GPU. See [GPU setup](#gpu-setup) if `pip install` can't get a vLLM wheel for your CUDA driver.

---

## Usage

### Parse a page (both steps)

```bash
indic-doc-parser parse page.png -o out/page
```

You get two files:
- **`out/page.md`** — the page as Markdown, in reading order
- **`out/page.json`** — one entry per block, with text ([details below](#output))

`page.png` can also be a **folder** — every image in it is parsed, and results are written into `-o`.

### Just the layout

```bash
indic-doc-parser layout page.png -o out/page      # -> out/page.layout.json
```

### Just the OCR — bring your own layout

```bash
indic-doc-parser ocr page.png -o out/page         # reads out/page.layout.json
```

Pass `--layout your_layout.json` to use a layout from somewhere else (any file in the
[layout format](#pagelayoutjson--the-intermediate)).

### From Python

```python
from indic_doc_parser import IndicDocParser

result = IndicDocParser().parse("page.png")
print(result["markdown"])
for block in result["blocks"]:
    print(block["order"], block["type"], block["text"])
```

Or the two steps separately:

```python
from indic_doc_parser import LayoutParser, BlockOCR

layout = LayoutParser().detect("page.png")      # step 1 — quick, no OCR model loaded
result = BlockOCR().run("page.png", layout)     # step 2 — `layout` can be your own
```

---

## Input

Anywhere a command takes an image you can pass **one image** (`.png`, `.jpg`, `.jpeg`, `.webp`,
`.bmp`, `.tif`) **or a folder of images** (processed recursively).

---

## Output

### `page.md`

The whole page as Markdown, blocks joined in reading order, with words hyphenated across a line break
rejoined. Math is LaTeX (`$…$` inline, `$$…$$` display); tables are Markdown tables.

```markdown
অধ্যায়—৬

ন্যায়নির্ণয়

২৮। ধাৰা ২৫ৰ অধীনত অভিগৃহীত কোনো সামগ্ৰী বা প্ৰচলিত মুদ্ৰা বা প্ৰতিভূতি বাজেয়াপ্তকৰণলৈ …
```

### `page.json`

One entry per block, in reading order:

```json
{
  "image": "page.png",
  "blocks": [
    {
      "order": 3,
      "type": "SectionHeader",
      "label": "Section-title",
      "bbox_xyxy": [844.9, 361.1, 997.9, 406.3],
      "conf": 0.784,
      "text": "অধ্যায়—৬"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `order` | reading-order position (`0` = first) |
| `type` | block type: `Text`, `SectionHeader`, `Title`, `ListItem`, `Table`, `Equation`, `Caption`, `Footnote` |
| `label` | the raw layout label, before it's simplified to `type` |
| `bbox_xyxy` | box on the page in pixels, `[x0, y0, x1, y1]` (top-left → bottom-right) |
| `conf` | how confident the layout model is, `0`–`1` |
| `text` | the transcribed content — LaTeX for math, Markdown for tables |

Page furniture — page numbers, running headers/footers, figures — is left out of the text.

### `page.layout.json` — the intermediate

What the **layout** step produces, before any OCR. Same shape, minus the `text`:

```json
{
  "image": "page.png",
  "width": 1653,
  "height": 2339,
  "blocks": [
    {"order": 1, "type": "Text", "label": "Paragraph", "bbox_xyxy": [120.0, 210.4, 1533.1, 388.7], "conf": 0.81}
  ]
}
```

Edit this, or produce it from your own layout model, and feed it to `indic-doc-parser ocr`.

---

## Languages

English and 22 Indian languages — Assamese, Bengali, Bodo, Dogri, Konkani, Gujarati, Hindi, Kannada,
Kashmiri, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi,
Tamil, Telugu, Urdu — across printed and handwritten pages.

---

## GPU setup

Everything runs on one GPU in one process. `pip install` pulls vLLM, which works on most CUDA
drivers. If your driver needs specific wheels, build the environment with the helper — it matches
wheels to your driver and checks the GPU before you download 1.7 GB of weights:

```bash
bash envs/setup_inference_env.sh          # creates a venv with CUDA-matched wheels
```

Verified stack: `vllm 0.26.1` · `torch 2.13.0+cu129` · `transformers 5.14.1`.

---

## Under the hood

Two small models:

- **Layout** — an RT-DETR detector with a built-in reading-order head (~33 M params), trained on
  192 k Indian-language pages across print and handwriting.
- **OCR** — a 0.8 B Qwen3.5 vision-language model (*Bodhan Scribe*) that reads each block. It uses the
  **Sarvam tokenizer** — a large, Indic-tuned vocabulary — which is the main reason it's a bit larger
  than English-only OCR models of a similar shape.

On **olmOCR-Bench (English)** the pipeline scores **82.9**, ahead of comparable open document parsers
on the same test. Its weak spot is complex multi-column reading order.

---

## License

The layout model derives from PP-DocLayoutV3 and the OCR model from the Qwen family; the respective
upstream licenses apply. The Private-Preview weights are additionally covered by your evaluation
agreement with Bodhan AI — access is limited to named evaluators, and the weights may not be
redistributed or used to train other models without written permission.

---

<sub>IndicDocParser-Private-Preview · © Bodhan AI · Confidential</sub>
