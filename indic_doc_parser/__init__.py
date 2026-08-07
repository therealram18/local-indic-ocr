"""IndicDocParser — modular block-level Indic document parser (layout + reading order + OCR)."""
from .pipeline import IndicDocParser, LayoutParser, BlockOCR

__all__ = ["IndicDocParser", "LayoutParser", "BlockOCR"]
__version__ = "0.1.0"
