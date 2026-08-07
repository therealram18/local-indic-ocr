"""CLI: `indic-doc-parser` — two stages that also compose.

  indic-doc-parser layout IMAGE… -o OUT     # stage 1: -> OUT.layout.json (blocks + reading order)
  indic-doc-parser ocr    IMAGE… -o OUT     # stage 2: OUT.layout.json + image -> OUT.md + OUT.json
  indic-doc-parser parse  IMAGE… -o OUT     # both stages in one process -> OUT.md + OUT.json
  indic-doc-parser show-contract            # prompts / block taxonomy / output schema
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
            sys.exit(f"not an image or directory: {item}")
    if not out:
        sys.exit("no input images found")
    return out


def stem_for(img, out, multi):
    """Output stem: OUT-dir/img-stem when multi/dir output, else OUT or the image path sans suffix."""
    if multi or (out and (Path(out).is_dir() or out.endswith("/"))):
        out_dir = Path(out) if out else img.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / img.stem)
    stem = out or str(img.with_suffix(""))
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    return stem


def write_layout(layout, stem):
    with open(stem + ".layout.json", "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)


def write_ocr(result, stem):
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"image": result["image"], "blocks": result["blocks"]}, f, ensure_ascii=False, indent=2)


def cmd_layout(args):
    from .pipeline import LayoutParser
    images = collect_images(args.images)
    lp = LayoutParser(layout_ckpt=args.layout_ckpt)
    for img in images:
        layout = lp.detect(str(img))
        stem = stem_for(img, args.out, len(images) > 1)
        write_layout(layout, stem)
        print(f"[layout] {img.name} -> {stem}.layout.json ({len(layout['blocks'])} blocks)", file=sys.stderr)


def cmd_ocr(args):
    from .pipeline import BlockOCR
    images = collect_images(args.images)
    oc = BlockOCR(ocr_ckpt=args.ocr_ckpt, gpu_memory_utilization=args.gpu_memory_utilization)
    for img in images:
        stem = stem_for(img, args.out, len(images) > 1)
        lay_path = args.layout or (stem + ".layout.json")   # where `layout -o <same>` wrote it
        if not os.path.isfile(lay_path):
            sys.exit(f"no layout json for {img.name}: {lay_path} (run `indic-doc-parser layout -o "
                     f"{stem}` first, or pass --layout)")
        with open(lay_path, encoding="utf-8") as f:
            layout = json.load(f)
        result = oc.run(str(img), layout)
        write_ocr(result, stem)
        print(f"[ocr] {img.name} -> {stem}.md ({len(result['markdown'])} chars) + "
              f"{stem}.json ({len(result['blocks'])} blocks)", file=sys.stderr)


def cmd_parse(args):
    from .pipeline import IndicDocParser
    images = collect_images(args.images)
    p = IndicDocParser(layout_ckpt=args.layout_ckpt, ocr_ckpt=args.ocr_ckpt,
                       gpu_memory_utilization=args.gpu_memory_utilization)
    for img in images:
        result = p.parse(str(img))
        stem = stem_for(img, args.out, len(images) > 1)
        write_ocr(result, stem)
        if args.save_layout:
            write_layout(result["layout"], stem)
        print(f"[parse] {img.name} -> {stem}.md ({len(result['markdown'])} chars) + "
              f"{stem}.json ({len(result['blocks'])} blocks)"
              + (f" + {stem}.layout.json" if args.save_layout else ""), file=sys.stderr)


def cmd_show_contract(_args):
    from . import contract as C
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


def _add_ckpt_args(sp, layout=False, ocr=False, gpu=False):
    if layout:
        sp.add_argument("--layout-ckpt", default=None, help="override layout checkpoint dir")
    if ocr:
        sp.add_argument("--ocr-ckpt", default=None, help="override OCR checkpoint dir")
    if gpu:
        sp.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                        help="vLLM gpu_memory_utilization")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="indic-doc-parser",
        description="Block-level Indic document parsing — layout+reading-order then per-block OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")

    pl = sub.add_parser("layout", help="stage 1: image -> layout.json (blocks + reading order)")
    pl.add_argument("images", nargs="+"); pl.add_argument("-o", "--out", default=None)
    _add_ckpt_args(pl, layout=True); pl.set_defaults(func=cmd_layout)

    po = sub.add_parser("ocr", help="stage 2: image + layout.json -> md + per-block json")
    po.add_argument("images", nargs="+")
    po.add_argument("--layout", default=None, help="layout.json (default: <image>.layout.json sidecar)")
    po.add_argument("-o", "--out", default=None)
    _add_ckpt_args(po, ocr=True, gpu=True); po.set_defaults(func=cmd_ocr)

    pp = sub.add_parser("parse", help="both stages in one process -> md + per-block json")
    pp.add_argument("images", nargs="+"); pp.add_argument("-o", "--out", default=None)
    pp.add_argument("--save-layout", action="store_true", help="also write OUT.layout.json")
    _add_ckpt_args(pp, layout=True, ocr=True, gpu=True); pp.set_defaults(func=cmd_parse)

    sub.add_parser("show-contract", help="print prompts / block taxonomy / output schema") \
        .set_defaults(func=cmd_show_contract)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help(); sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
