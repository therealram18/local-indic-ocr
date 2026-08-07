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

# IndicDocParser (Private Preview)

Give IndicDocParser a page image, in English or any of 22 Indian languages, and it returns the page
as Markdown plus a per-block JSON. Math comes back as LaTeX and tables as Markdown.

It runs in two steps:

1. **Layout.** Find the blocks on the page (paragraphs, headings, tables, equations, and so on) and
   put them in reading order.
2. **OCR.** Read the text inside each block.

You can run both steps at once, or run them separately. Running them separately lets you take the
layout on its own, or supply your own layout and run only the OCR.

> **Confidential, Private Preview.** These weights are shared for evaluation only. Please do not
> redistribute them or publish outputs attributed to the model without written permission from Bodhan AI.

## Install

```bash
pip install indic-doc-parser
```

The weights (about 1.85 GB) download automatically the first time you run the tool. They are private,
so log in to Hugging Face once first:

```bash
huggingface-cli login
```

You need one GPU. If `pip install` cannot find a vLLM wheel for your CUDA driver, see [GPU setup](#gpu-setup).

## Quick start

```bash
indic-doc-parser parse page.png -o out
```

This writes `out/page.md` and `out/page.json`. In place of `page.png` you can pass a folder, and every
image in it is processed.

## Two steps

Stage 1 produces a layout file. Stage 2 reads that layout file and does the OCR. The layout file is
the only thing passed between the steps, and it records the image it came from, so stage 2 takes just
the layout.

```bash
# stage 1: detect blocks and reading order
indic-doc-parser layout page.png -o out
#   writes out/page.layout.json

# stage 2: OCR each block in that layout
indic-doc-parser ocr out/page.layout.json -o out
#   reads out/page.layout.json (and the image it names), writes out/page.md and out/page.json
```

If the image has moved since stage 1, point stage 2 at it with `--image`:

```bash
indic-doc-parser ocr out/page.layout.json --image page.png -o out
```

## From Python

```python
from indic_doc_parser import IndicDocParser

result = IndicDocParser().parse("page.png")
print(result["markdown"])
for block in result["blocks"]:
    print(block["order"], block["type"], block["text"])
```

The two steps are separate objects. Stage 2 accepts any layout in the format below, including one you
made yourself.

```python
from indic_doc_parser import LayoutParser, BlockOCR

layout = LayoutParser().detect("page.png")     # stage 1, returns the layout dict
result = BlockOCR().run("page.png", layout)    # stage 2, takes that layout
```

## Input

Every command takes one of:

* a single image file (`png`, `jpg`, `jpeg`, `webp`, `bmp`, `tif`), or
* a folder of images (all images inside are processed).

Outputs are named after each input image and written to the folder given by `-o` (by default, next to
the image).

## Output

Three kinds of file, all named after the input image.

**`page.layout.json`** (from `layout`, or from `parse --save-layout`). The blocks on the page and
their reading order. No text yet.

```json
{
  "image": "page.png",
  "image_path": "/data/pages/page.png",
  "width": 1653,
  "height": 2339,
  "blocks": [
    {"order": 1, "type": "Text", "label": "Paragraph", "bbox_xyxy": [120.0, 210.4, 1533.1, 388.7], "conf": 0.81}
  ]
}
```

`image_path` is where the image was when the layout was made, so `ocr` can find it. Pass `--image`
to override it.

**`page.md`** (from `ocr`, or from `parse`). The page as Markdown, blocks joined in reading order,
with words split across a line break rejoined. Math is LaTeX (`$...$` inline, `$$...$$` display) and
tables are Markdown tables.

**`page.json`** (from `ocr`, or from `parse`). Every block, in reading order, with its text.

```json
{
  "image": "page.png",
  "blocks": [
    {"order": 3, "type": "SectionHeader", "label": "Section-title", "bbox_xyxy": [844.9, 361.1, 997.9, 406.3], "conf": 0.784, "text": "..."}
  ]
}
```

Each block has these fields:

| Field | Meaning |
|---|---|
| `order` | position in reading order, starting at 0 |
| `type` | block type: `Text`, `SectionHeader`, `Title`, `ListItem`, `Table`, `Equation`, `Caption`, `Footnote` |
| `label` | the original layout label, before it is simplified to `type` |
| `bbox_xyxy` | box on the page in pixels, `[x0, y0, x1, y1]`, from the top-left to the bottom-right corner |
| `conf` | how confident the layout model is, from 0 to 1 |
| `text` | the transcription (LaTeX for math, Markdown for tables). Present in `page.json` only |

Page furniture, such as page numbers, running headers and footers, and figures, is left out of the text.

## Languages

English and 22 Indian languages: Assamese, Bengali, Bodo, Dogri, Konkani, Gujarati, Hindi, Kannada,
Kashmiri, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi,
Tamil, Telugu, and Urdu. Both printed and handwritten pages are supported.

## GPU setup

Everything runs on one GPU in a single process. `pip install` pulls vLLM, which works on most CUDA
drivers. If your driver needs specific wheels, build the environment with the helper script, which
matches wheels to your driver and checks the GPU before downloading the weights:

```bash
bash envs/setup_inference_env.sh
```

Verified stack: vLLM 0.26.1, PyTorch 2.13.0 (cu129), Transformers 5.14.1.

## Under the hood

Two small models:

* **Layout.** An RT-DETR detector with a built-in reading-order head, about 33M parameters, trained on
  192k Indian-language pages spanning print and handwriting.
* **OCR.** A 0.8B Qwen3.5 vision-language model (Bodhan Scribe) that reads each block. It uses the
  Sarvam tokenizer, a large vocabulary tuned for Indian scripts. That large vocabulary is the main
  reason the model is a little larger than English-only OCR models of a similar shape.

On olmOCR-Bench (English) the pipeline scores 82.9, ahead of comparable open document parsers on the
same test. Its main weakness is complex multi-column reading order.

## License

The layout model derives from PP-DocLayoutV3 and the OCR model from the Qwen family, so the upstream
licenses apply. The Private-Preview weights are also covered by your evaluation agreement with Bodhan
AI: access is limited to named evaluators, and the weights may not be redistributed or used to train
other models without written permission.
