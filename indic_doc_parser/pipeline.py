"""IndicDocParser — block-level document parsing: layout+reading-order harness -> per-block OCR.
One process, one GPU, one env (vLLM venv: transformers + vLLM). Produces markdown + per-block JSON.
"""
import os
import re

from . import config
config.apply()                      # set env defaults BEFORE ocr_lib import (it reads them at import)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from . import ocr_lib as OI
from .layout.infer import get_model, infer

# Checkpoint resolution: prefer weights bundled in the repo (weights/{layout,ocr}, present after a
# `huggingface-cli download oikantik-bodhan/indic-doc-parser`), else fall back to the dev locations.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent of indic_doc_parser/


def _resolve_ckpt(sub, dev_path):
    bundled = os.path.join(_REPO_ROOT, "weights", sub)
    return bundled if os.path.isdir(bundled) else dev_path


LAYOUT_CKPT_DEFAULT = _resolve_ckpt("layout", "/projects/data/visionteam/oikantik/harness/training/best/final")
OCR_CKPT_DEFAULT = _resolve_ckpt("ocr", "/projects/data/visionteam/oikantik/deploy-tests/sft-v6-ckpt")

_HYPH = re.compile(r'(\w)[\-‐‑‒–—−]\n[ \t]*(\w)')


def dehyph(t: str) -> str:
    """Join words split by a hyphen at a line break."""
    prev = None
    while prev != t:
        prev = t
        t = _HYPH.sub(r'\1\2', t)
    return t


class IndicDocParser:
    def __init__(self, layout_ckpt=LAYOUT_CKPT_DEFAULT, ocr_ckpt=OCR_CKPT_DEFAULT,
                 device="cuda", gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION):
        # vLLM FIRST: its EngineCore must init CUDA before the torch layout model touches the device.
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        self.proc = AutoProcessor.from_pretrained(ocr_ckpt)
        self.llm = LLM(model=ocr_ckpt, trust_remote_code=True, max_model_len=OI.MAX_MODEL_LEN,
                       gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",
                       limit_mm_per_prompt={"image": 1}, enforce_eager=True)
        self.sp = SamplingParams(temperature=OI.TEMPERATURE, max_tokens=OI.MAX_TOKENS)
        self.lm = get_model(layout_ckpt, model_type="ppdoc", device=device)
        self.device = device

    def _chat(self, prompt):
        return self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=False)

    def parse(self, image_path):
        """Parse one page image -> {'image', 'markdown', 'blocks'[{order,type,label,bbox_xyxy,conf,text}]}."""
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        key = os.path.basename(image_path)

        # 1) layout + reading order -> pixel [x0,y0,x1,y1] entries for the harness oracle
        dets = infer(self.lm, img, conf=config.LAYOUT_CONF, img_size=config.LAYOUT_IMG_SIZE,
                     device=self.device)                          # bbox [y0,x0,y1,x1]/1000
        layout = {key: [{"bbox": [d["bbox"][1] / 1000 * W, d["bbox"][0] / 1000 * H,
                                  d["bbox"][3] / 1000 * W, d["bbox"][2] / 1000 * H],
                         "label": d["label"], "order": d["reading_order"], "conf": d["score"]}
                        for d in dets]}
        blocks = OI.detect_layout_harness(layout, key, img)       # furniture-drop + dedup + crops

        # 2) per-block OCR (Table/Equation/text prompts via prompt_for_type)
        tb = OI.ocr_blocks(self.llm, self.sp, self._chat, {key: blocks}).get(key, {})

        # 3) markdown (dehyphenated) + per-block json
        md = dehyph(OI.reconstruct(blocks, tb))
        records = [{"order": b["order"], "type": b["type"], "label": b.get("label"),
                    "bbox_xyxy": [round(float(v), 1) for v in b["bbox"]],
                    "conf": round(float(b.get("conf", 1.0)), 3),
                    "text": (tb.get(b["id"]) or "").strip()}
                   for b in sorted(blocks, key=lambda x: x["order"])]
        return {"image": key, "markdown": md, "blocks": records}
