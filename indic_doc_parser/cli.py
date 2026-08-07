"""CLI:  indic-doc-parser <image> [-o out_stem]   ->  writes out_stem.md + out_stem.json"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(prog="indic-doc-parser",
                                 description="Block-level Indic document parsing (layout+RO + OCR).")
    ap.add_argument("image", help="page image (png/jpg)")
    ap.add_argument("-o", "--out", default=None, help="output stem (default: alongside the image)")
    ap.add_argument("--layout-ckpt", default=None, help="override layout checkpoint dir")
    ap.add_argument("--ocr-ckpt", default=None, help="override OCR checkpoint dir")
    ap.add_argument("--gpu-mem", type=float, default=0.80, help="vLLM gpu_memory_utilization")
    args = ap.parse_args()

    from .pipeline import IndicDocParser
    kw = {"gpu_memory_utilization": args.gpu_mem}
    if args.layout_ckpt:
        kw["layout_ckpt"] = args.layout_ckpt
    if args.ocr_ckpt:
        kw["ocr_ckpt"] = args.ocr_ckpt

    parser = IndicDocParser(**kw)
    result = parser.parse(args.image)

    stem = args.out or os.path.splitext(args.image)[0]
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"image": result["image"], "blocks": result["blocks"]}, f,
                  ensure_ascii=False, indent=2)
    print(f"-> {stem}.md ({len(result['markdown'])} chars) + {stem}.json ({len(result['blocks'])} blocks)")


if __name__ == "__main__":
    main()
