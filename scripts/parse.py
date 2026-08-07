#!/usr/bin/env python3
"""IndicDocParser -- block-level document parsing (layout + reading order -> per-block OCR).

Runs the whole pipeline in one process on one GPU: the layout+reading-order model detects
and orders blocks, the recognizer transcribes each, and the output is stitched back into
reading-ordered markdown plus a per-block JSON. No server, no conversion step.

The prompts, block taxonomy, and output schema live in ``idp_contract.py``.

Examples
--------
    # one page (weights are auto-resolved from ./weights next to this repo)
    python scripts/parse.py page.png -o out/page

    # a directory of pages, on GPU 2
    CUDA_VISIBLE_DEVICES=2 python scripts/parse.py pages/ -o out/

    # point at a downloaded repo / different checkpoints explicitly
    python scripts/parse.py --model /path/to/indic-doc-parser page.png -o out/page

Output
------
    <out>.md   -- reconstructed, reading-ordered, dehyphenated markdown
    <out>.json -- {"image", "blocks":[{order,type,label,bbox_xyxy,conf,text}, …]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# vLLM stability flags for this checkpoint (set before the engine imports).
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent          # repo root (holds indic_doc_parser/ + weights/)
sys.path.insert(0, str(REPO_ROOT))                          # import the engine without `pip install`

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
        else:
            sys.exit(f"not an image or directory: {item}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse document pages into reading-ordered markdown + per-block JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("images", nargs="*", help="page image(s) and/or directorie(s) of images")
    ap.add_argument("--model", default=str(REPO_ROOT),
                    help="repo root or HF snapshot dir holding weights/{layout,ocr}")
    ap.add_argument("-o", "--out", default=None,
                    help="output stem (single image) or output directory (multiple)")
    ap.add_argument("--show-contract", action="store_true",
                    help="print the prompts / block taxonomy / output schema and exit")

    eng = ap.add_argument_group("engine")
    eng.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    eng.add_argument("--min-px-side", type=int, default=None,
                     help="crop-upscale floor (default 256; overrides OCR_MIN_PX_SIDE)")
    eng.add_argument("--dedup-mode", default=None,
                     help="nested-equation dedup mode (default text_only)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import idp_contract as C

    if args.show_contract:
        print("PROMPTS:")
        print("  text/heading :", C.OCR_PROMPT)
        print("  Table        :", C.TABLE_PROMPT)
        print("  Equation     :", C.EQUATION_PROMPT)
        print("\nKEPT BLOCK TYPES:", ", ".join(C.KEPT_BLOCK_TYPES))
        print("DROPPED (furniture):", ", ".join(C.FURNITURE_LABELS))
        print("\nOUTPUT <name>.json blocks[i]:")
        for k, v in C.OUTPUT_BLOCK_SCHEMA.items():
            print(f"  {k:<10} {v}")
        print("\nDEFAULT RECIPE:", C.DEFAULTS)
        return

    if not args.images:
        ap.error("pass at least one image or directory (or --show-contract)")

    # config overrides must be set before the engine imports (ocr_lib reads them at import)
    if args.min_px_side is not None:
        os.environ["OCR_MIN_PX_SIDE"] = str(args.min_px_side)
    if args.dedup_mode is not None:
        os.environ["DEDUP_MODE"] = args.dedup_mode

    images = collect_images(args.images)
    if not images:
        sys.exit("no input images found")

    model_root = Path(args.model)
    kw = {"gpu_memory_utilization": args.gpu_memory_utilization}
    if (model_root / "weights" / "layout").is_dir():
        kw["layout_ckpt"] = str(model_root / "weights" / "layout")
        kw["ocr_ckpt"] = str(model_root / "weights" / "ocr")

    from indic_doc_parser import IndicDocParser
    print(f"[run] {len(images)} image(s); loading engine …", file=sys.stderr)
    parser = IndicDocParser(**kw)

    multi = len(images) > 1
    for img in images:
        result = parser.parse(str(img))
        if multi or (args.out and (Path(args.out).is_dir() or args.out.endswith("/"))):
            out_dir = Path(args.out) if args.out else img.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = str(out_dir / img.stem)
        else:
            stem = args.out or str(img.with_suffix(""))
            d = os.path.dirname(stem)
            if d:
                os.makedirs(d, exist_ok=True)
        with open(stem + ".md", "w", encoding="utf-8") as f:
            f.write(result["markdown"])
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump({"image": result["image"], "blocks": result["blocks"]}, f,
                      ensure_ascii=False, indent=2)
        print(f"[ok] {img.name} -> {stem}.md ({len(result['markdown'])} chars) + "
              f"{stem}.json ({len(result['blocks'])} blocks)", file=sys.stderr)


if __name__ == "__main__":
    main()
