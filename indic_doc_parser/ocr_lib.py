#!/usr/bin/env python3
"""ocr_infer.py — Reference inference pipeline: page PNG -> markdown.

Deployment-reference implementation of the validated block-OCR stack
(olmOCR-bench 80.6, best of a 7-oracle bake-off, 2026-07):

    PNG page image
      |
      v
    [1] LAYOUT   plus-l (default) | v3 | surya  -- see ORACLE NOTES below
      |             blocks: bbox + class + reading order
      |             plus-l: pages that return no boxes at 0.50 retry once at
      |             0.20, then fall back to OCR'ing the whole page as one block
      |             (archival handwriting would otherwise emit blank markdown)
      v
    [2] OCR      Qwen-08B checkpoint via vLLM, one request per block:
      |             * crops smart-upscaled (aspect-preserving AREA clamp
      |               into [448^2, 1536^2] px -> <=~2304 image tokens, never
      |               exceeds context; small crops gain resolution: +1.1 bench)
      |             * type-aware prompts (Table->GFM, Equation->LaTeX,
      |               else plain transcription)
      |             * page-furniture blocks are never OCR'd (dropped)
      v
    [3] ASSEMBLE drop furniture, sort by reading order, wrap bare equations in
      |          display-math delimiters, join with "\n\n"
      v
    markdown

ORACLE NOTES (bench scores are with smart-upscale, English subset):
    plus-l (DEFAULT, WINNER, 80.6)  PaddleOCR RT-DETR, Apache-2.0. Column-aware
        geometric reading order (beats every learned reading-order head we
        tested). Adaptive retry + full-page fallback (above).
    v3     (79.8)  PP-DocLayoutV3, transformers-native (Apache-2.0-derived
        PaddleOCR license), RT-DETR + built-in reading-order decoder. Keeps its
        OWN native decoder order -- do NOT re-sort. Weaker on multi-column than
        plus-L; reportedly better on handwriting. No adaptive-retry path.
    surya  (78.3)  Surya-2 layout model. *** OpenRAIL-M license -- NOT
        license-clean for closed-source commercial shipping; included ONLY for
        side-by-side comparison/research. Requires `pip install surya-ocr` (not
        in the validated .venv_ref). Uses Surya's own native block order. ***

DISABLED here (present but off; enable if a change needs them, then re-bench):
adjacent-table merging and paddle's containment `layout_merge_bboxes_mode`.

Both the OCR model and every layout oracle fit ONE GPU. vLLM serves the OCR
model (continuous batching: submit every block of every page in a single
generate() call). No layout oracle here is vLLM-servable (all are object-
detection nets) -- each runs via its own engine (paddle / transformers /
surya-ocr) in the same process (~0.1-0.3 s/page, negligible next to OCR).

Dependencies (validated versions; plus-l/v3 come from setup_env.sh):
    vllm                >= 0.9        (eval used 0.20.1)
    paddlepaddle-gpu    == 3.0.0      \\_ only needed for --layout-oracle plus-l
    paddleocr           == 3.7.0      /
    transformers        (pulled by vllm; also runs --layout-oracle v3)
    surya-ocr                         # only needed for --layout-oracle surya
    pillow

Usage:
    python ocr_infer.py --images page1.png page2.png --out-dir out/
    python ocr_infer.py --images scans/ --out-dir out/        # every *.png in dir
    python ocr_infer.py --images scans/ --out-dir out/ --layout-oracle v3
    python ocr_infer.py --images scans/ --out-dir out/ --layout-oracle surya

Serving note: for an online service, replace the offline `LLM()` engine with
vLLM's OpenAI-compatible server or AsyncLLMEngine and keep stages [1]+[3] in
the request handler; per-block requests preserve the batching benefit.

Input assumption: page images at roughly 150-300 dpi equivalent (the pipeline
was validated on 200 dpi renders). The crop area-clamp absorbs most resolution
variation; extremely low-res inputs degrade OCR itself, not the plumbing.
"""

import argparse
import math
import os
import sys
from pathlib import Path

# MUST be set before vllm import. paddlex registers a broken vllm plugin when
# both share a venv (VLLM_PLUGINS="" skips it), and if any library initialises
# CUDA in this parent process, vLLM's default forked EngineCore dies with
# "Cannot re-initialize CUDA in forked subprocess" -> spawn.
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None      # some bench pages render to >178M px at high DPI (legit large
#                                    scans/newspapers); load them — layout resizes to 1024 and
#                                    crops are area-clamped, so compute stays bounded.

# ---------------------------------------------------------------------------
# Constants (all values validated on olmOCR-bench; change with care)
# ---------------------------------------------------------------------------

# Block classes that are detected but NEVER OCR'd nor emitted (running headers/footers/page
# numbers are noise on olmOCR-bench; figures/pictures have no OCR-able text -> hallucination).
DROP_TYPES = {"PageHeader", "PageFooter", "PageNumber", "Figure", "Picture"}

# Type-aware prompts.
OCR_PROMPT = os.environ.get("OCR_PROMPT", "Transcribe the text in this image.")  # env-overridable for prompt A/Bs
TABLE_PROMPT = ("Convert this table image to a GitHub-flavored markdown table. "
                "Output only the table.")
EQUATION_PROMPT = "Output only the LaTeX for this equation image."
# For text blocks that CONTAIN inline equations (DEDUP_MODE=relabel): transcribe prose AND render the
# math as LaTeX, so the ONE paragraph box replaces both the malformed base-prompt text and the nested
# equation box (which we then drop). The model does inline math well when told to expect it.
MATH_TEXT_PROMPT = os.environ.get("MATH_TEXT_PROMPT",
    "Transcribe all visible text exactly as shown, preserving characters, casing, punctuation, and "
    "spacing. Render every mathematical expression in LaTeX, using $...$ for inline math and $$...$$ "
    "for display equations. Do not translate, correct, or omit anything.")

# Smart crop upscale: aspect-preserving AREA clamp. Pinning a *side* instead
# explodes elongated crops (a 122:1 rule line -> 32k tokens -> engine wedge);
# clamping AREA bounds worst-case image tokens at MAX_PX/32^2 ~= 2304 for any
# aspect ratio.
# OCR_MIN_PX_SIDE tunes the small-crop upscale floor; set to 0 to DISABLE upscaling (only downscale
# large crops) — LANCZOS upscaling blurs degraded old-scan math glyphs and can drop fine marks, so
# disabling it is an old_scans_math-specific lever (sherry's crops are not upscaled).
MIN_PX = int(os.environ.get("OCR_MIN_PX_SIDE", "448")) ** 2   # crops below this area are upscaled (default: +1.1)
MAX_PX = int(os.environ.get("OCR_MAX_PX_SIDE", "1536")) ** 2   # crops above this area are downscaled (token ceiling)
CROP_PAD_PX = int(os.environ.get("CROP_PAD_PX", "0"))         # px margin added around each bbox before cropping
                                                             # (recovers glyph edges our tight boxes clip)

# Adjacent-table merge: same column band (x-overlap >= 50%) and vertical gap
# < 40 px at ~200 dpi (~2 mm).
TABLE_MERGE_MAX_GAP_PX = 40
TABLE_MERGE_MIN_XOVERLAP = 0.5

# vLLM engine settings (validated).
MAX_MODEL_LEN = 8192
MAX_TOKENS = 2048       # per-block generation cap
OCR_BATCH = 2048        # max requests per llm.generate() call (a single huge batch wedges vLLM V1)
TEMPERATURE = 0.0       # deterministic transcription

LAYOUT_MODEL_NAME = "PP-DocLayout_plus-L"
SAME_LABEL_CONTAINMENT = 0.98

# V3 default: our internal, dtype-cast-validated safetensors copy. For a
# standalone deployment without access to /projects/data, pass
# --v3-model PaddlePaddle/PP-DocLayoutV3 (public HF repo; untested by us --
# requires a transformers version with native PP-DocLayoutV3 support).
V3_MODEL_DEFAULT = "/projects/data/visionteam/hf_cache/PP-DocLayoutV3_safetensors"
V3_THRESHOLD_DEFAULT = 0.3

# Adaptive layout fallback. The model's shipped draw threshold is 0.50. Pages
# outside its printed-document domain (notably archival handwriting) can return
# no boxes, which previously skipped OCR entirely and silently emitted blank
# markdown. Retry once at 0.20; do not chase progressively lower thresholds,
# where stains/rules/borders become arbitrary detections. If the retry still
# has no meaningful OCR-able coverage, OCR the full page as one Text block.
LAYOUT_THRESHOLD = 0.50
# Disabled baseline experiments (kept for later comparison):
LAYOUT_DEFAULT_THRESHOLD = 0.50
LAYOUT_RETRY_THRESHOLD = 0.20
LAYOUT_MIN_USABLE_COVERAGE = 0.01

# Containment suppression (paddle-native). layout_nms kills near-duplicates but
# NOT nested boxes (containment = low IoU); without paddle's assembly-stage text
# -cell dedup (which we don't have — we OCR each crop independently), a box
# contained in another box double-emits its text. Fix: paddle's documented
# layout_merge_bboxes_mode, per plus-L category index: "large" = drop boxes
# contained (>=90%) inside another box. Base dict = PP-StructureV3's shipped
# config; text/reference-family entries added because our pipeline lacks their
# downstream dedup. Formulas inside text are hard-protected by paddle (never
# suppressed) so math granularity is unaffected. Verified: removes exactly the
# duplicated-reference boxes on the test pages, zero change elsewhere.
LAYOUT_MERGE_BBOXES_MODE = {
    0: "large",   # paragraph_title   (PP-StructureV3)
    2: "large",   # text              (added: no assembly dedup in our pipeline)
    7: "large",   # formula           (PP-StructureV3)
    9: "large",   # reference         (added)
    19: "large",  # reference_content (added)
}

# PP-DocLayout_plus-L label -> pipeline block type. Only the DROP_TYPES,
# "Table" and "Equation" change behaviour; every other label gets the plain
# OCR prompt. Unknown labels fall through to "Text".
PADDLE_TO_TYPE = {
    "doc_title": "Title", "title": "Title", "paragraph_title": "SectionHeader",
    "text": "Text", "abstract": "Text", "content": "Text", "reference": "Text",
    "algorithm": "Text", "aside_text": "Text", "formula_number": "Text",
    "footnote": "Footnote",
    "figure_title": "Caption", "chart_title": "Caption", "table_title": "Caption",
    "formula": "Equation", "display_formula": "Equation",
    "inline_formula": "Equation", "interline_formula": "Equation",
    "table": "Table",
    # page furniture / non-text -> dropped
    "header": "PageHeader", "footer": "PageFooter",
    "number": "PageNumber", "page_number": "PageNumber",
    "header_image": "PageHeader", "footer_image": "PageFooter",
    "image": "Figure", "figure": "Figure", "chart": "Figure", "seal": "Picture",
}


def map_label(paddle_label) -> str:
    key = str(paddle_label or "").strip().lower().replace(" ", "_")
    return PADDLE_TO_TYPE.get(key, "Text")


# PP-DocLayoutV3 label -> pipeline block type (25-label vocab; distinct from
# plus-L's, e.g. V3 has no "table_title"/"chart_title" but does have
# "reference_content" as its own category).
V3_TO_TYPE = {
    "formula": "Equation", "table": "Table",
    "header": "PageHeader", "footer": "PageFooter", "number": "PageNumber",
    "footnote": "Footnote", "vision_footnote": "Footnote",
    "doc_title": "Title", "paragraph_title": "SectionHeader", "figure_title": "Caption",
    "text": "Text", "abstract": "Text", "content": "Text", "reference": "Text",
    "reference_content": "Text", "aside_text": "Text", "algorithm": "Text",
    "formula_number": "Text",
    "chart": "Figure", "image": "Picture", "seal": "Picture",
}


def map_label_v3(v3_label) -> str:
    return V3_TO_TYPE.get(str(v3_label).strip().lower(), "Text")


# Harness layout+RO model label (37-class superset) -> pipeline block type. Only Table /
# Equation change the OCR prompt; everything text-like -> Text. Furniture kept (DROP_TYPES
# is empty) so it still OCRs. Keyed by the lowercased class name from harness src/labels.py.
HARNESS_TO_TYPE = {
    "table": "Table", "table-caption": "Caption",
    "equation": "Equation", "expression": "Equation",
    "diagram": "Figure", "chart": "Figure", "image": "Picture", "image-caption": "Caption",
    "title": "Title", "chapter-title": "Title",
    "section-title": "SectionHeader", "sub-section-title": "SectionHeader",
    "sub-sub-section-title": "SectionHeader",
    "header": "PageHeader", "footer": "PageFooter",
    "page-number": "PageNumber", "folio": "PageNumber",
    "footnote": "Footnote",
    # text-like (default): question, paragraph, answer, list, mcq, infobox, placeholder-text,
    # solved-example, code, chapter-end-section, reference, table-of-contents, index,
    # advertisement, author, dateline, contact-info, website-link, flag
}


def map_label_harness(label) -> str:
    return HARNESS_TO_TYPE.get(str(label).strip().lower(), "Text")


# Harness (37-class) labels we NEVER OCR — decided on OUR taxonomy, not paddle's DROP_TYPES:
# figures/pictures have no text; running header/footer/page-number/folio are furniture; and
# Advertisement is treated as furniture too. (Flag + Placeholder-text are KEPT as text.)
HARNESS_DROP_LABELS = {"diagram", "image", "chart", "header", "footer",
                       "page-number", "folio", "advertisement"}

# Nested-box de-duplication. The harness over-produces Equation boxes: small inline-math boxes
# nested INSIDE a text paragraph, and per-line boxes nested inside a display-equation array. When
# both the container and the nested boxes are OCR'd and concatenated by reading order, the math is
# emitted twice (inline-in-text AND as a standalone $$ block) and fragmented -> breaks olmo math
# grouping tests (arxiv_math / old_scans_math). Rule: drop an Equation box that sits >=70% inside a
# strictly-larger (a) text-like block -> inline math is left to the container's OCR; or (b) Equation
# block -> a multi-row array stays ONE block. Toggle with DEDUP_NEST=0.
DEDUP_NEST = os.environ.get("DEDUP_NEST", "1") == "1"
# DEDUP_MODE: which containers can absorb a nested Equation.
#   "both"      -> text-like OR Equation (rule 1 + rule 2)                       [default]
#   "eq_only"   -> Equation only          (rule 2: collapse display arrays; leave inline-in-text)  [Variant A]
#   "text_only" -> text-like only         (rule 1)
#   "tight"     -> Equation always (rule 2) + text-like ONLY IF container area <= DEDUP_TIGHT_RATIO x eqn area
#   "density"   -> Equation always (rule 2) + text-like ONLY IF the container is PROSE-DOMINANT, i.e. its   [Variant B]
#                  equation coverage <= DEDUP_EQ_COVER. Drops sparse inline math (arxiv) but SPARES a
#                  formula-list box whose equations fill most of it (the old_scans_math regression).
#   "count"     -> Equation always (rule 2) + text-like ONLY IF it has FEW equation children
#                  (<= DEDUP_MAX_CHILDREN): sparse inline math (arxiv) is deduped; a formula LIST
#                  with many stacked equation boxes (old_scans_math) is spared. Count separates the
#                  two cases better than area-coverage (numbers/whitespace dilute a list's coverage).
_TEXTLIKE = {"Text", "Title", "SectionHeader", "Caption", "Footnote"}
DEDUP_MODE = os.environ.get("DEDUP_MODE", "both")
DEDUP_TIGHT_RATIO = float(os.environ.get("DEDUP_TIGHT_RATIO", "3.0"))
DEDUP_EQ_COVER = float(os.environ.get("DEDUP_EQ_COVER", "0.5"))
DEDUP_MAX_CHILDREN = int(os.environ.get("DEDUP_MAX_CHILDREN", "4"))
DEDUP_MAX_WIDTH_FRAC = float(os.environ.get("DEDUP_MAX_WIDTH_FRAC", "0.6"))  # "width" mode: drop inner eqn only if narrower than this fraction of container width


def _drop_nested_inner(inner, container, gate):
    """inner is an Equation already >=70% contained in container. Decide (per DEDUP_MODE) whether to
    drop it. `gate` = container's equation coverage (density) or equation-child count (count)."""
    ct = container["type"]
    if ct == "Equation":
        return DEDUP_MODE != "text_only"                       # rule 2 (collapse arrays)
    if ct in _TEXTLIKE:
        if DEDUP_MODE == "eq_only":
            return False
        if DEDUP_MODE == "tight":
            return _box_area(container["bbox"]) <= DEDUP_TIGHT_RATIO * _box_area(inner["bbox"])
        if DEDUP_MODE == "density":
            return gate <= DEDUP_EQ_COVER
        if DEDUP_MODE == "count":
            return gate <= DEDUP_MAX_CHILDREN
        if DEDUP_MODE == "width":
            # drop only a NARROW inner eqn (true inline fragment); spare a wide one that spans the
            # line (an old-scan problem equation that needs equation-OCR, not text-OCR).
            cw = container["bbox"][2] - container["bbox"][0]
            iw = inner["bbox"][2] - inner["bbox"][0]
            return cw > 0 and (iw / cw) < DEDUP_MAX_WIDTH_FRAC
        return True                                            # both / text_only (rule 1)
    return False


def _box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contained_in(inner, outer, thr=0.70):
    ia = _box_area(inner)
    if ia <= 0:
        return False
    iw = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (iw * ih) / ia >= thr and ia < 0.95 * _box_area(outer)


def _container_eq_cover(container, blocks):
    """Fraction of container area covered by Equation boxes >=70% contained in it (sum-approx,
    capped 1.0). High => container is really an equation/formula region, not prose."""
    ca = _box_area(container["bbox"])
    if ca <= 0:
        return 0.0
    s = sum(_box_area(b["bbox"]) for b in blocks
            if b is not container and b["type"] == "Equation" and _contained_in(b["bbox"], container["bbox"]))
    return min(1.0, s / ca)


def _container_eq_count(container, blocks):
    """Number of Equation boxes >=70% contained in the container. High => a formula LIST, not prose."""
    return sum(1 for b in blocks
               if b is not container and b["type"] == "Equation" and _contained_in(b["bbox"], container["bbox"]))


def resolve_nested_equations(blocks: list) -> list:
    """Drop each Equation block that is >=70% contained in a strictly-larger text-like or Equation
    block, per DEDUP_MODE (see note). Keeps the outer container; removes the redundant inner equation."""
    gate = {}
    if DEDUP_MODE == "density":
        gate = {j: _container_eq_cover(c, blocks) for j, c in enumerate(blocks) if c["type"] in _TEXTLIKE}
    elif DEDUP_MODE == "count":
        gate = {j: _container_eq_count(c, blocks) for j, c in enumerate(blocks) if c["type"] in _TEXTLIKE}
    drop = set()
    for i, e in enumerate(blocks):
        if e["type"] != "Equation":
            continue
        for j, c in enumerate(blocks):
            if i == j:
                continue
            if _contained_in(e["bbox"], c["bbox"]) and _drop_nested_inner(e, c, gate.get(j, 0.0)):
                drop.add(i)
                # relabel mode: the text container ABSORBS this equation -> OCR it math-aware so the
                # inline LaTeX is rendered correctly (replacing both the dropped eqn box and the
                # malformed base-prompt version). One clean box, nothing held back.
                if DEDUP_MODE == "relabel" and c["type"] in _TEXTLIKE:
                    c["math_aware"] = True
                break
    return [b for k, b in enumerate(blocks) if k not in drop]


def detect_layout_harness(layout_by_stem: dict, stem: str, img: Image.Image) -> list:
    """Build oracle blocks from a PRECOMPUTED harness layout JSON (produced out-of-process by
    olmocr_harness/harness_layout.py in the harness venv). Entries are pixel [x0,y0,x1,y1] +
    our reading-order (`order`); rank follows that order so reconstruct() uses OUR RO head."""
    W, H = img.size
    entries = sorted(layout_by_stem.get(stem, []), key=lambda d: d.get("order", 0))
    blocks = []
    for rank, e in enumerate(entries):
        x0, y0, x1, y1 = (float(v) for v in e["bbox"])
        bbox = [max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)]
        btype = map_label_harness(e.get("label"))
        blocks.append(_make_oracle_block(rank, bbox, btype, e.get("label"),
                                         float(e.get("conf", 1.0)), img, W, H, "harness"))
    # Filter (harness taxonomy): (1) drop HARNESS_DROP_LABELS (figures / running furniture / ads);
    # (2) drop ANY block whose box sits >=70% inside a Header/Footer region — running-header/footer
    # content and page numbers/small text captured in those bands are noise on olmOCR-bench.
    hf = [b["bbox"] for b in blocks if str(b["label"]).strip().lower() in ("header", "footer")]

    def _inside_hf(bb):
        ba = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
        if ba <= 0:
            return False
        for h in hf:
            iw = max(0.0, min(bb[2], h[2]) - max(bb[0], h[0]))
            ih = max(0.0, min(bb[3], h[3]) - max(bb[1], h[1]))
            if iw * ih / ba >= 0.70:
                return True
        return False

    kept = [b for b in blocks
            if str(b["label"]).strip().lower() not in HARNESS_DROP_LABELS and not _inside_hf(b["bbox"])]
    if DEDUP_NEST:
        kept = resolve_nested_equations(kept)
    return kept


def prompt_for_type(block_type: str) -> str:
    if block_type == "Table":
        return TABLE_PROMPT
    if block_type == "Equation":
        return EQUATION_PROMPT
    return OCR_PROMPT


# ---------------------------------------------------------------------------
# [1] Layout: detection + reading order
# ---------------------------------------------------------------------------

def geometric_reading_order(boxes: list) -> list:
    """Column-aware reading order: cluster boxes into columns by x-centre,
    order columns left->right, within a column top->bottom. Deterministic.
    (Outperformed both learned reading-order heads we tested.)"""
    n = len(boxes)
    if n <= 1:
        return list(range(n))

    def xc(b):
        x1, _, x2, _ = b["coordinate"]
        return (x1 + x2) / 2.0

    widths = sorted(b["coordinate"][2] - b["coordinate"][0] for b in boxes)
    medw = widths[len(widths) // 2] or 1.0
    tol = medw * 0.5                      # half a median width = same column

    columns = []                          # {"c": running centre, "items": [idx]}
    for idx in sorted(range(n), key=lambda i: xc(boxes[i])):
        cx = xc(boxes[idx])
        for col in columns:
            if abs(cx - col["c"]) < tol:
                col["items"].append(idx)
                col["c"] = sum(xc(boxes[j]) for j in col["items"]) / len(col["items"])
                break
        else:
            columns.append({"c": cx, "items": [idx]})

    columns.sort(key=lambda col: col["c"])
    order = []
    for col in columns:
        order.extend(sorted(col["items"], key=lambda i: boxes[i]["coordinate"][1]))
    return order


def extract_boxes(res) -> list:
    """Normalise a PaddleOCR LayoutDetection result to [{coordinate, label, score}]."""
    if hasattr(res, "json"):
        res = res.json
    if isinstance(res, dict):
        for key in ("res", "result"):
            if key in res and isinstance(res[key], dict):
                res = res[key]
        if "boxes" in res:
            return list(res["boxes"])
    raise ValueError("could not locate 'boxes' in LayoutDetection result")


def _predict_layout_boxes(layout_model, bgr: np.ndarray, threshold: float) -> list:
    results = layout_model.predict(
        bgr,
        batch_size=1,
        threshold=threshold,
        layout_nms=True,
        # BASELINE: custom containment suppression disabled.
        # layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,
    )
    res = results[0] if isinstance(results, (list, tuple)) else results
    return extract_boxes(res)


def suppress_contained_same_label(boxes: list,
                                  containment: float = SAME_LABEL_CONTAINMENT) -> list:
    """Drop a smaller box almost fully contained by a larger same-label box.

    This deliberately does not compare different Paddle labels. A page-sized
    `image` can therefore never suppress `text`, and a `paragraph_title` inside
    a text region remains independently OCR-able.
    """
    def area(box):
        x1, y1, x2, y2 = box["coordinate"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def contained_fraction(inner, outer):
        ix1, iy1, ix2, iy2 = inner["coordinate"]
        ox1, oy1, ox2, oy2 = outer["coordinate"]
        intersection = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(
            0.0, min(iy2, oy2) - max(iy1, oy1)
        )
        return intersection / max(1.0, area(inner))

    ranked = sorted(
        enumerate(boxes),
        key=lambda item: (-area(item[1]), -float(item[1].get("score", 0.0)), item[0]),
    )
    kept = []
    for original_index, candidate in ranked:
        label = str(candidate.get("label") or "").strip().lower()
        if any(
            label == str(outer.get("label") or "").strip().lower()
            and contained_fraction(candidate, outer) >= containment
            for _, outer in kept
        ):
            continue
        kept.append((original_index, candidate))
    return [box for _, box in sorted(kept, key=lambda item: item[0])]


def _usable_layout_coverage(boxes: list, width: int, height: int) -> float:
    """Conservative summed coverage used only to reject near-empty retries."""
    page_area = max(1.0, float(width * height))
    area = 0.0
    for box in boxes:
        if map_label(box.get("label")) in DROP_TYPES:
            continue
        x1, y1, x2, y2 = box["coordinate"]
        clipped_w = max(0.0, min(float(width), x2) - max(0.0, x1))
        clipped_h = max(0.0, min(float(height), y2) - max(0.0, y1))
        area += clipped_w * clipped_h
    return min(1.0, area / page_area)


def _layout_is_usable(boxes: list, width: int, height: int) -> bool:
    """Reject layouts with no meaningful OCR-able page coverage."""
    return bool(boxes) and _usable_layout_coverage(
        boxes, width, height
    ) >= LAYOUT_MIN_USABLE_COVERAGE


def detect_layout(layout_model, img: Image.Image,
                  threshold: float = LAYOUT_THRESHOLD) -> list:
    """Run plus-L on a PIL page image -> ordered block list
    [{id, bbox, type, order, crop(PIL)}] (furniture types carry no crop)."""
    W, H = img.size
    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()   # paddle wants BGR
    boxes = _predict_layout_boxes(layout_model, bgr, threshold)
    boxes = suppress_contained_same_label(boxes)
    layout_strategy = f"baseline_{threshold:.2f}"

    # Adaptive fallback: pages outside plus-L's printed-document domain
    # (notably archival handwriting) can return no usable boxes at 0.50, which
    # would otherwise skip OCR entirely and silently emit blank markdown. Retry
    # once at 0.20; if still empty, OCR the whole page as one Text block.
    if not _layout_is_usable(boxes, W, H):
        boxes = _predict_layout_boxes(layout_model, bgr, LAYOUT_RETRY_THRESHOLD)
        boxes = suppress_contained_same_label(boxes)
        layout_strategy = "retry_0.20"
    if not _layout_is_usable(boxes, W, H):
        return [{
            "id": 0, "bbox": [0.0, 0.0, float(W), float(H)],
            "type": "Text", "label": "full_page_fallback", "order": 0,
            "conf": 0.0, "crop": img.copy().convert("RGB"),
            "layout_strategy": "full_page_fallback",
        }]

    order = geometric_reading_order(boxes)
    blocks = []
    for rank, bidx in enumerate(order):
        b = boxes[bidx]
        x1, y1, x2, y2 = b["coordinate"]
        bbox = [max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)]
        paddle_label = str(b.get("label") or "unknown")
        btype = map_label(paddle_label)
        crop = None
        if btype not in DROP_TYPES:
            cx0, cy0, cx1, cy1 = (int(round(v)) for v in bbox)
            if cx1 > cx0 and cy1 > cy0:     # int-rounding can zero a thin box
                crop = img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        blocks.append({"id": rank, "bbox": bbox, "type": btype,
                       "label": paddle_label,
                       "order": rank, "conf": float(b.get("score", 1.0)),
                       "crop": crop, "layout_strategy": layout_strategy})
    return blocks


def _make_oracle_block(rank, bbox, btype, label, conf, img, W, H, strategy):
    """Per-block crop/dict construction for the v3 and surya oracles. Keeps the
    same schema plus-L emits (id/bbox/type/label/order/conf/crop/layout_strategy)
    so engine.py and the web viewer stay oracle-agnostic. Furniture carries no
    crop."""
    crop = None
    if btype not in DROP_TYPES:
        cx0, cy0, cx1, cy1 = (int(round(v)) for v in bbox)
        if CROP_PAD_PX:                     # margin to recover glyph edges tight boxes clip
            cx0 -= CROP_PAD_PX; cy0 -= CROP_PAD_PX; cx1 += CROP_PAD_PX; cy1 += CROP_PAD_PX
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(W, cx1), min(H, cy1)
        if cx1 > cx0 and cy1 > cy0:         # int-rounding can zero a thin box
            crop = img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    return {"id": rank, "bbox": [float(v) for v in bbox], "type": btype,
            "label": str(label), "order": rank, "conf": float(conf),
            "crop": crop, "layout_strategy": strategy}


def detect_layout_v3(model, proc, img: Image.Image, threshold: float) -> list:
    """PP-DocLayoutV3 oracle (79.8). Keeps V3's OWN native decoder order --
    do NOT re-sort; every alternative order we tried scored worse."""
    import torch
    inputs = proc(images=[img], return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    for k in list(outputs.keys()):
        v = outputs[k]
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            outputs[k] = v.float()   # post_process_object_detection can't handle bf16

    W, H = img.size
    result = proc.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[H, W]]), threshold=threshold)[0]
    labels = [model.config.id2label[label_id.item()] for label_id in result["labels"]]

    blocks = []
    for rank, (lab, box, sc) in enumerate(zip(labels, result["boxes"], result["scores"])):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        bbox = [max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)]
        btype = map_label_v3(lab)
        blocks.append(_make_oracle_block(rank, bbox, btype, lab, sc.item(),
                                         img, W, H, "v3"))
    return blocks


def detect_layout_surya(layout_pred, img: Image.Image) -> list:
    """Surya-2 oracle (78.3). *** OpenRAIL-M -- comparison/research only,
    do not ship. *** Surya's own label vocabulary IS this pipeline's `type`
    vocabulary (we adopted it), so no label map is needed. Reading order:
    Surya's own `.position` field (native; not re-sorted, matching the
    original validated harness)."""
    W, H = img.size
    layout = layout_pred([img])[0]
    indexed = sorted(enumerate(layout.bboxes), key=lambda x: x[1].position)

    blocks = []
    for rank, (_, lb) in enumerate(indexed):
        poly = lb.polygon
        if poly and len(poly) >= 2:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            bbox = [0.0, 0.0, float(W), float(H)]
        btype = str(lb.label)
        blocks.append(_make_oracle_block(rank, bbox, btype, lb.label,
                                         lb.confidence or 0.0, img, W, H, "surya"))
    return blocks


def suppress_contained_blocks(blocks: list, containment: float = 0.90) -> list:
    """Drop any box whose area is >= `containment` inside a LARGER box, keeping
    the larger (parent) box. Label-agnostic, on finished block dicts.

    Prefers page coverage over granularity: a page-spanning parent is kept and
    its fully-covered children are dropped, so no content slips through an
    uncovered gap between children. Trade-off — a page-sized parent then OCRs as
    one coarse crop (downscaled, generic prompt); if that hurts, switch to a
    coverage-guard hybrid. Returns survivors in original order.
    """
    def area(block):
        x1, y1, x2, y2 = block["bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def contained_fraction(inner, outer):
        ix1, iy1, ix2, iy2 = inner["bbox"]
        ox1, oy1, ox2, oy2 = outer["bbox"]
        intersection = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(
            0.0, min(iy2, oy2) - max(iy1, oy1)
        )
        return intersection / max(1.0, area(inner))

    ranked = sorted(
        enumerate(blocks),
        key=lambda item: (-area(item[1]), -float(item[1].get("conf", 0.0)), item[0]),
    )
    kept = []
    for original_index, candidate in ranked:
        if any(contained_fraction(candidate, outer) >= containment
               for _, outer in kept):
            continue
        kept.append((original_index, candidate))
    return [block for _, block in sorted(kept, key=lambda item: item[0])]


# ---------------------------------------------------------------------------
# [2] OCR: crop conditioning + per-block vLLM requests
# ---------------------------------------------------------------------------

def area_clamp(img: Image.Image) -> Image.Image:
    """Aspect-preserving area clamp into [MIN_PX, MAX_PX]."""
    w, h = img.size
    area = w * h
    if area <= 0:
        return img
    if area < MIN_PX:
        s = math.sqrt(MIN_PX / area)
    elif area > MAX_PX:
        s = math.sqrt(MAX_PX / area)
    else:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def _x_overlap_frac(b1, b2):
    lo, hi = max(b1[0], b2[0]), min(b1[2], b2[2])
    overlap = max(0.0, hi - lo)
    s1, s2 = b1[2] - b1[0], b2[2] - b2[0]
    if s1 <= 0 or s2 <= 0:
        return 0.0
    return min(overlap / s1, overlap / s2)


def merge_adjacent_tables(blocks: list) -> list:
    """Merge runs of consecutive Table blocks that share a column band and are
    vertically contiguous, by stacking their crops -> one prompt sees the
    whole table instead of fragments."""
    ordered = sorted(blocks, key=lambda b: b["order"])
    result, i = [], 0
    while i < len(ordered):
        blk = ordered[i]
        if blk["type"] != "Table" or blk["crop"] is None:
            result.append(blk)
            i += 1
            continue
        run, j = [blk], i + 1
        while j < len(ordered):
            nxt, prev = ordered[j], run[-1]
            if (nxt["type"] == "Table" and nxt["crop"] is not None
                    and _x_overlap_frac(prev["bbox"], nxt["bbox"]) >= TABLE_MERGE_MIN_XOVERLAP
                    and nxt["bbox"][1] - prev["bbox"][3] < TABLE_MERGE_MAX_GAP_PX):
                run.append(nxt)
                j += 1
            else:
                break
        if len(run) > 1:
            crops = [b["crop"] for b in run]
            merged = Image.new("RGB", (max(c.width for c in crops),
                                       sum(c.height for c in crops)), (255, 255, 255))
            y = 0
            for c in crops:
                merged.paste(c, (0, y))
                y += c.height
            head = dict(run[0])
            head["crop"] = merged
            head["bbox"] = [min(b["bbox"][0] for b in run), run[0]["bbox"][1],
                            max(b["bbox"][2] for b in run), run[-1]["bbox"][3]]
            result.append(head)
        else:
            result.append(blk)
        i = j if len(run) > 1 else i + 1
    return result


def ocr_blocks(llm, sampling_params, chat_template_fn, pages: dict) -> dict:
    """OCR every retained block of every page in ONE vLLM generate() call.

    pages: {page_key: [block, ...]}  ->  {page_key: {block_id: text}}
    """
    requests, keys = [], []
    for page_key, blocks in pages.items():
        for b in blocks:
            if b["type"] in DROP_TYPES or b["crop"] is None:
                continue
            # relabel mode: a text block that absorbed an inline equation is OCR'd with the equation
            # prompt (sft-v6 was TRAINED on paragraph+inline-LaTeX with this prompt, so it transcribes
            # prose AND renders math). RELABEL_PROMPT=mathtext switches to the explicit transcribe prompt.
            if b.get("math_aware"):
                _mp = MATH_TEXT_PROMPT if os.environ.get("RELABEL_PROMPT") == "mathtext" else EQUATION_PROMPT
                _prompt = chat_template_fn(_mp)
            else:
                _prompt = chat_template_fn(prompt_for_type(b["type"]))
            requests.append({
                "prompt": _prompt,
                "multi_modal_data": {"image": area_clamp(b["crop"])},
            })
            keys.append((page_key, b["id"]))

    texts = {}
    # Submit in bounded sub-batches. One giant generate() over tens of thousands of
    # multi-modal requests wedges the vLLM V1 scheduler (100% util, no progress); ~2k-request
    # chunks run cleanly and let big page-sets (e.g. a 1258-page benchmark) complete.
    for i in range(0, len(requests), OCR_BATCH):
        sub, subkeys = requests[i:i + OCR_BATCH], keys[i:i + OCR_BATCH]
        outputs = llm.generate(sub, sampling_params)        # continuous batching within the chunk
        for (page_key, bid), out in zip(subkeys, outputs):
            texts.setdefault(page_key, {})[bid] = out.outputs[0].text
    return texts


# ---------------------------------------------------------------------------
# [3] Assemble
# ---------------------------------------------------------------------------

def reconstruct(blocks: list, text_by_id: dict) -> str:
    """Drop furniture, sort by reading order, join non-empty texts."""
    kept = [b for b in blocks if b["type"] not in DROP_TYPES]
    kept.sort(key=lambda b: b["order"])
    parts = []
    for b in kept:
        text = (text_by_id.get(b["id"]) or "").strip()
        if not text:
            continue
        # Equation blocks whose OCR text arrives without math delimiters would
        # render as literal LaTeX source; wrap bare equations in display math.
        if b["type"] == "Equation" and not (text.startswith("$") or
                text.startswith("\\[") or text.startswith("\\(")):
            text = f"$${text}$$"
        parts.append(text)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(paths: list) -> list:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.glob("*.png")))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Page PNG -> markdown (selectable layout oracle + Qwen vLLM OCR)")
    ap.add_argument("--images", nargs="+", required=True,
                    help="PNG file(s) and/or directorie(s) of PNGs")
    ap.add_argument("--ckpt", required=True, help="Consolidated Qwen checkpoint dir")
    ap.add_argument("--out-dir", required=True, help="Output dir for per-page .md files")
    ap.add_argument("--layout-json", default=None,
                    help="precomputed harness layout JSON (required for --layout-oracle harness)")
    ap.add_argument("--layout-oracle", choices=["plus-l", "v3", "surya", "harness"], default="plus-l",
                    help="Layout detector. plus-l = WINNER/default, 80.6, Apache-2.0. "
                         "v3 = 79.8, Apache-2.0-derived, native reading order kept as-is. "
                         "surya = 78.3, OpenRAIL-M -- NOT license-clean for closed-source "
                         "shipping; comparison/research use only, requires `pip install "
                         "surya-ocr` separately.")
    ap.add_argument("--v3-model", default=V3_MODEL_DEFAULT,
                    help="V3 model id/path (only used with --layout-oracle v3)")
    ap.add_argument("--v3-threshold", type=float, default=V3_THRESHOLD_DEFAULT,
                    help="V3 detection confidence threshold (only used with --layout-oracle v3)")
    ap.add_argument("--gpu-mem", type=float, default=0.80,
                    help="vLLM gpu_memory_utilization (leave headroom for the layout oracle)")
    ap.add_argument("--layout-threshold", type=float, default=LAYOUT_THRESHOLD,
                    help="plus-l detection threshold (default 0.50; only used with --layout-oracle plus-l)")
    args = ap.parse_args()

    images = collect_images(args.images)
    if not images:
        sys.exit("no input images found")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- fail fast on a missing oracle dependency, BEFORE the ~2-4 min vLLM
    # engine startup (a late ImportError here would waste that entire cost) --
    import importlib.util
    if args.layout_oracle == "plus-l":
        if importlib.util.find_spec("paddleocr") is None:
            sys.exit("--layout-oracle plus-l requires `pip install paddleocr paddlepaddle-gpu` "
                     "(see SETUP.md)")
    elif args.layout_oracle == "v3":
        if importlib.util.find_spec("transformers") is None:
            sys.exit("--layout-oracle v3 requires `transformers` (already a vllm dependency)")
    elif args.layout_oracle == "harness":
        if not args.layout_json or not os.path.isfile(args.layout_json):
            sys.exit("--layout-oracle harness requires --layout-json <path> to a precomputed "
                     "layout JSON (produce it with olmocr_harness/harness_layout.py)")
    else:  # surya
        if importlib.util.find_spec("surya") is None:
            sys.exit("--layout-oracle surya requires `pip install surya-ocr` (not in the "
                     "validated .venv_ref -- and NOTE: Surya-2 is OpenRAIL-M licensed, not "
                     "license-clean for closed-source commercial shipping; comparison/research "
                     "use only)")

    # --- load both engines once ------------------------------------------
    # ORDER MATTERS: construct the vLLM engine BEFORE paddle touches CUDA.
    # vLLM forks/spawns its EngineCore at LLM() time; if paddle has already
    # initialised CUDA in this process, a forked child cannot re-init it.
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(args.ckpt)

    def chat_template_fn(prompt_text):
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        return processor.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)

    llm = LLM(model=args.ckpt, trust_remote_code=True, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              limit_mm_per_prompt={"image": 1},
              enforce_eager=os.environ.get("OCR_EAGER") == "1")   # skip ~4min torch.compile (0.8B model)
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # --- load the selected layout oracle AFTER vLLM (CUDA-init ordering) --
    # vLLM forks/spawns its EngineCore at LLM() time; if a layout oracle
    # (paddle, torch/transformers, or surya-ocr's own torch model) has already
    # initialised CUDA in this process, the fork cannot re-init it.
    v3_model = v3_proc = layout_model = layout_pred = harness_layout = None
    if args.layout_oracle == "plus-l":
        from paddleocr import LayoutDetection
        layout_model = LayoutDetection(model_name=LAYOUT_MODEL_NAME)
    elif args.layout_oracle == "v3":
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        v3_proc = AutoImageProcessor.from_pretrained(args.v3_model)
        v3_model = AutoModelForObjectDetection.from_pretrained(
            args.v3_model, dtype=torch.bfloat16).cuda().eval()
    elif args.layout_oracle == "harness":
        import json as _json
        harness_layout = _json.load(open(args.layout_json))   # {stem: [blocks]}; no GPU model here
        print(f"[layout:harness] loaded precomputed layout for {len(harness_layout)} pages "
              f"from {args.layout_json}", flush=True)
    else:  # surya
        from surya.layout import LayoutPredictor
        layout_pred = LayoutPredictor()

    # --- [1] layout for all pages ----------------------------------------
    pages = {}
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:            # corrupt/unreadable input -> empty page, don't crash the batch
            pages[str(img_path)] = []
            print(f"[layout:{args.layout_oracle}] {img_path.name}: SKIP unreadable ({e})", flush=True)
            continue
        if args.layout_oracle == "plus-l":
            blocks = detect_layout(layout_model, img, args.layout_threshold)
        elif args.layout_oracle == "v3":
            blocks = detect_layout_v3(v3_model, v3_proc, img, threshold=args.v3_threshold)
        elif args.layout_oracle == "harness":
            blocks = detect_layout_harness(harness_layout, img_path.stem, img)
        else:
            blocks = detect_layout_surya(layout_pred, img)
        # BASELINE: adjacent-table merging disabled.
        # blocks = merge_adjacent_tables(blocks)
        pages[str(img_path)] = blocks
        print(f"[layout:{args.layout_oracle}] {img_path.name}: {len(blocks)} blocks", flush=True)

    # --- [2] OCR all blocks of all pages in one batched call --------------
    texts = ocr_blocks(llm, sampling, chat_template_fn, pages)

    # --- [3] assemble + write ---------------------------------------------
    for img_path in images:
        key = str(img_path)
        md = reconstruct(pages[key], texts.get(key, {}))
        out_path = Path(args.out_dir) / (img_path.stem + ".md")
        out_path.write_text(md, encoding="utf-8")
        print(f"[done] {out_path}  ({len(md)} chars)", flush=True)


if __name__ == "__main__":
    main()
