"""Re-export of the canonical contract (now in the installed package)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indic_doc_parser.contract import *   # noqa: F401,F403
from indic_doc_parser.contract import (   # explicit for linters
    TABLE_PROMPT, EQUATION_PROMPT, OCR_PROMPT, KEPT_BLOCK_TYPES,
    FURNITURE_LABELS, OUTPUT_BLOCK_SCHEMA, DEFAULTS,
)
