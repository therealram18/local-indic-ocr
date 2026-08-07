"""CLI: `indic-doc-parser`. Two stages that also compose.

  indic-doc-parser layout IMG... -o DIR                    stage 1: writes DIR/<name>.layout.json
  indic-doc-parser ocr    IMG... -o DIR                    stage 2: reads DIR/<name>.layout.json,
                                                                     writes DIR/<name>.md + .json
  indic-doc-parser parse  IMG... -o DIR                    both stages in one process
  indic-doc-parser show-contract                           prompts, block types, output fields

Stage 1 produces the layout; stage 2 consumes it. Point stage 2 at a specific layout file with
--layout, or it looks for <name>.layout.json in the output directory.
"""
import argparse
import json
import os
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(inputs):
    out = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
        else:
            sys.exit(f"not an image or a folder: {item}")
    if not out:
        sys.exit("no input images found")
    return out


def out_path(img, out_dir, suffix, mkdir=False):
    """Output path for one image: <out_dir or image's folder>/<image name><suffix>."""
    d = Path(out_dir) if out_dir else img.parent
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return str(d / (img.stem + suffix))


def cmd_layout(args):
    from .pipeline import LayoutParser
    lp = LayoutParser(layout_ckpt=args.layout_ckpt)
    for img in collect_images(args.images):
        layout = lp.detect(str(img))
        dst = out_path(img, args.out_dir, ".layout.json", mkdir=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(layout, f, ensure_ascii=False, indent=2)
        print(f"[layout] {img.name}: {len(layout['blocks'])} blocks -> {dst}", file=sys.stderr)


def collect_layouts(inputs):
    out = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out += sorted(p.rglob("*.layout.json"))
        elif p.suffix == ".json":
            out.append(p)
        else:
            sys.exit(f"not a layout .json or a folder: {item}")
    if not out:
        sys.exit("no layout files found")
    return out


def cmd_ocr(args):
    from .pipeline import BlockOCR
    oc = BlockOCR(ocr_ckpt=args.ocr_ckpt, gpu_memory_utilization=args.gpu_memory_utilization)
    for lay_path in collect_layouts(args.layouts):
        with open(lay_path, encoding="utf-8") as f:
            layout = json.load(f)
        img_path = args.image or layout.get("image_path")
        if not img_path or not os.path.isfile(img_path):
            sys.exit(f"image for {lay_path.name} not found ({img_path}); pass --image <file>")
        result = oc.run(img_path, layout)
        out_dir = args.out_dir or str(lay_path.parent)        # default: next to the layout file
        img = Path(layout.get("image") or os.path.basename(img_path))
        write_ocr(result, img, out_dir)


def cmd_parse(args):
    from .pipeline import IndicDocParser
    p = IndicDocParser(layout_ckpt=args.layout_ckpt, ocr_ckpt=args.ocr_ckpt,
                       gpu_memory_utilization=args.gpu_memory_utilization)
    for img in collect_images(args.images):
        result = p.parse(str(img))
        if args.save_layout:
            dst = out_path(img, args.out_dir, ".layout.json", mkdir=True)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(result["layout"], f, ensure_ascii=False, indent=2)
        write_ocr(result, img, args.out_dir)


def write_ocr(result, img, out_dir):
    md = out_path(img, out_dir, ".md", mkdir=True)
    js = out_path(img, out_dir, ".json")
    with open(md, "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"image": result["image"], "blocks": result["blocks"]}, f, ensure_ascii=False, indent=2)
    print(f"[ocr] {img.name}: {len(result['blocks'])} blocks -> {md}, {js}", file=sys.stderr)


def cmd_show_contract(_args):
    from . import contract as C
    print("PROMPTS")
    print("  text, headings:", C.OCR_PROMPT)
    print("  tables        :", C.TABLE_PROMPT)
    print("  equations     :", C.EQUATION_PROMPT)
    print("\nBLOCK TYPES kept:", ", ".join(C.KEPT_BLOCK_TYPES))
    print("DROPPED as furniture:", ", ".join(C.FURNITURE_LABELS))
    print("\nOUTPUT json, each block:")
    for k, v in C.OUTPUT_BLOCK_SCHEMA.items():
        print(f"  {k:<10} {v}")
    print("\nDEFAULTS:", C.DEFAULTS)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="indic-doc-parser",
        description="Block-level Indic document parsing: layout and reading order, then per-block OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")

    def add_io(sp):
        sp.add_argument("images", nargs="+", help="an image, or a folder of images")
        sp.add_argument("-o", "--out-dir", default=None,
                        help="output folder (default: next to each input image)")

    pl = sub.add_parser("layout", help="stage 1: image to layout.json (blocks and reading order)")
    add_io(pl); pl.add_argument("--layout-ckpt", default=None, help="override the layout checkpoint")
    pl.set_defaults(func=cmd_layout)

    po = sub.add_parser("ocr", help="stage 2: layout.json (from stage 1) to markdown and json")
    po.add_argument("layouts", nargs="+", help="layout.json file(s) from `layout`, or a folder of them")
    po.add_argument("-o", "--out-dir", default=None, help="output folder (default: next to the layout file)")
    po.add_argument("--image", default=None, help="override the image path recorded in the layout")
    po.add_argument("--ocr-ckpt", default=None, help="override the OCR checkpoint")
    po.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    po.set_defaults(func=cmd_ocr)

    pp = sub.add_parser("parse", help="both stages in one process")
    add_io(pp)
    pp.add_argument("--save-layout", action="store_true", help="also write <name>.layout.json")
    pp.add_argument("--layout-ckpt", default=None); pp.add_argument("--ocr-ckpt", default=None)
    pp.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    pp.set_defaults(func=cmd_parse)

    sub.add_parser("show-contract", help="print prompts, block types, and output fields") \
        .set_defaults(func=cmd_show_contract)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help(); sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
