"""IndicDocParser -- the pipeline contract.

Single source of truth for the block taxonomy, the per-block prompts, and the output
schema. Imported by ``parse.py`` so the CLI and any downstream consumer never drift from
what the engine actually produces. The live runtime values (prompts, dedup mode, crop
floor) are read by the engine from environment variables at import time -- see
``indic_doc_parser/config.py``; the constants here document the defaults and the contract.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Per-block prompts (what the recognizer is asked, per detected block type)
# --------------------------------------------------------------------------- #
#: Standalone Table blocks -> GitHub-flavored markdown.
TABLE_PROMPT = "Convert this table image to a GitHub-flavored markdown table. Output only the table."
#: Standalone Equation blocks -> LaTeX.
EQUATION_PROMPT = "Output only the LaTeX for this equation image."
#: Every other (text/paragraph/heading/…) block. Math-aware: inline $…$, display $$…$$.
OCR_PROMPT = ("Transcribe the text in this image. Write any mathematical expressions in "
              "LaTeX, using $...$ for inline math and $$...$$ for display equations.")

# --------------------------------------------------------------------------- #
# Block taxonomy
# --------------------------------------------------------------------------- #
#: Block types the layout model emits and the recognizer transcribes into the output.
KEPT_BLOCK_TYPES = [
    "Text", "SectionHeader", "Title", "ListItem", "Table", "Equation", "Caption", "Footnote",
]
#: Labels dropped as running furniture (never transcribed): page decoration, figures, ads.
FURNITURE_LABELS = ["diagram", "image", "chart", "header", "footer", "page-number", "folio"]

# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #
#: `<name>.md`   -- reconstructed, reading-ordered, dehyphenated markdown for the page.
#: `<name>.json` -- {"image": str, "blocks": [BLOCK, …]} in reading order, where BLOCK is:
OUTPUT_BLOCK_SCHEMA = {
    "order": "int   -- reading-order rank (0 = first)",
    "type": "str   -- block type (see KEPT_BLOCK_TYPES)",
    "label": "str   -- the raw layout label",
    "bbox_xyxy": "[float,float,float,float] -- pixel box [x0, y0, x1, y1]",
    "conf": "float -- layout detection confidence",
    "text": "str   -- transcribed content (LaTeX for math, markdown for tables)",
}

# --------------------------------------------------------------------------- #
# Default recipe (olmOCR-Bench-English "clean" config, 82.9). Override via env.
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "OCR_MIN_PX_SIDE": "256",     # crop-upscale floor (largest single win)
    "DEDUP_MODE": "text_only",    # drop inline equation boxes (the math prompt covers them)
    "DEDUP_NEST": "1",
    "LAYOUT_CONF": "0.5",
    "LAYOUT_IMG_SIZE": "1024",
}
