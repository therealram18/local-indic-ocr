#!/usr/bin/env python3
"""Thin wrapper for the "download repo + run script" path — delegates to the package CLI.

  python scripts/parse.py page.png -o out/page          # defaults to the `parse` subcommand
  python scripts/parse.py layout page.png -o out/page    # or an explicit stage
  python scripts/parse.py ocr    page.png -o out/page --layout out/page.layout.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # import the package without install
from indic_doc_parser.cli import main

SUBCOMMANDS = {"layout", "ocr", "parse", "show-contract"}
if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["parse"] + argv                                   # default to the combined stage
    main(argv)
