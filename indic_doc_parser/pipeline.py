"""IndicDocParser engine — two stages that also compose.

  Stage 1  LayoutParser : image           -> layout (blocks + reading order, no OCR).  torch only.
  Stage 2  BlockOCR     : image + layout  -> markdown + per-block text.               vLLM.
  Combined IndicDocParser: image          -> markdown + per-block text.               both models.

The stage-1 `layout` dict is the intermediate: {"image","width","height","blocks":[{order,type,
label,bbox_xyxy,conf}]}. Feed it straight into stage 2, or swap in your own layout of the same shape.
"""
import os
import re

from . import config
config.apply()                      # set env defaults BEFORE ocr_lib import (it reads them at import)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from . import ocr_lib as OI
from .layout.infer import get_model, infer

# Checkpoint resolution (lazy, at construction — never at import):
#   weights bundled in the repo  ->  local dev path  ->  auto-download from the HF repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent of indic_doc_parser/
HF_REPO = os.environ.get("INDIC_DOC_PARSER_REPO", "oikantik-bodhan/indic-doc-parser")
_DEV_PATHS = {
    "layout": "/projects/data/visionteam/oikantik/harness/training/best/final",
    "ocr": "/projects/data/visionteam/oikantik/deploy-tests/sft-v6-ckpt",
}


def resolve_ckpt(sub):
    """Return a local dir for weights/<sub>, downloading from the HF repo on first use if needed.
    (The HF repo is private — first download needs `huggingface-cli login` or HF_TOKEN.)"""
    bundled = os.path.join(_REPO_ROOT, "weights", sub)
    if os.path.isdir(bundled):
        return bundled
    if os.path.isdir(_DEV_PATHS.get(sub, "")):
        return _DEV_PATHS[sub]
    from huggingface_hub import snapshot_download
    root = snapshot_download(HF_REPO, allow_patterns=[f"weights/{sub}/*"])
    return os.path.join(root, "weights", sub)


_HYPH = re.compile(r'(\w)[\-‐‑‒–—−]\n[ \t]*(\w)')


def dehyph(t: str) -> str:
    """Join words split by a hyphen at a line break."""
    prev = None
    while prev != t:
        prev = t
        t = _HYPH.sub(r'\1\2', t)
    return t


class LayoutParser:
    """Stage 1 — layout + reading order. Loads only the layout model (no vLLM)."""

    def __init__(self, layout_ckpt=None, device="cuda"):
        self.lm = get_model(layout_ckpt or resolve_ckpt("layout"), model_type="ppdoc", device=device)
        self.device = device

    def detect(self, image_path):
        """image -> {"image","width","height","blocks":[{order,type,label,bbox_xyxy,conf}]}."""
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        dets = infer(self.lm, img, conf=config.LAYOUT_CONF, img_size=config.LAYOUT_IMG_SIZE,
                     device=self.device)                          # bbox [y0,x0,y1,x1]/1000
        blocks = [{"order": d["reading_order"],
                   "type": OI.map_label_harness(d["label"]),
                   "label": d["label"],
                   "bbox_xyxy": [round(d["bbox"][1] / 1000 * W, 1), round(d["bbox"][0] / 1000 * H, 1),
                                 round(d["bbox"][3] / 1000 * W, 1), round(d["bbox"][2] / 1000 * H, 1)],
                   "conf": round(float(d["score"]), 3)}
                  for d in dets]
        blocks.sort(key=lambda b: b["order"])
        return {"image": os.path.basename(image_path), "width": W, "height": H, "blocks": blocks}


class BlockOCR:
    """Stage 2 — per-block OCR given a layout. Loads the vLLM-served recognizer."""

    def __init__(self, ocr_ckpt=None, gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        ocr_ckpt = ocr_ckpt or resolve_ckpt("ocr")
        self.proc = AutoProcessor.from_pretrained(ocr_ckpt)
        self.llm = LLM(model=ocr_ckpt, trust_remote_code=True, max_model_len=OI.MAX_MODEL_LEN,
                       gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",
                       limit_mm_per_prompt={"image": 1}, enforce_eager=True)
        self.sp = SamplingParams(temperature=OI.TEMPERATURE, max_tokens=OI.MAX_TOKENS)

    def _chat(self, prompt):
        return self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=False)

    def run(self, image_path, layout):
        """image + stage-1 layout -> {"image","markdown","blocks":[{…,text}]}."""
        img = Image.open(image_path).convert("RGB")
        key = layout.get("image") or os.path.basename(image_path)
        entries = [{"bbox": b["bbox_xyxy"], "label": b["label"], "order": b["order"], "conf": b["conf"]}
                   for b in layout["blocks"]]
        blocks = OI.detect_layout_harness({key: entries}, key, img)   # furniture-drop + dedup + crops
        tb = OI.ocr_blocks(self.llm, self.sp, self._chat, {key: blocks}).get(key, {})
        md = dehyph(OI.reconstruct(blocks, tb))
        records = [{"order": b["order"], "type": b["type"], "label": b.get("label"),
                    "bbox_xyxy": [round(float(v), 1) for v in b["bbox"]],
                    "conf": round(float(b.get("conf", 1.0)), 3),
                    "text": (tb.get(b["id"]) or "").strip()}
                   for b in sorted(blocks, key=lambda x: x["order"])]
        return {"image": key, "markdown": md, "blocks": records}


class IndicDocParser:
    """Combined — both stages in one process (loads vLLM first, then the layout model)."""

    def __init__(self, layout_ckpt=None, ocr_ckpt=None, device="cuda",
                 gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION):
        # vLLM FIRST: its EngineCore must init CUDA before the torch layout model touches the device.
        self.ocr = BlockOCR(ocr_ckpt, gpu_memory_utilization=gpu_memory_utilization)
        self.layout = LayoutParser(layout_ckpt, device=device)

    def detect(self, image_path):
        """Stage 1 only -> layout dict."""
        return self.layout.detect(image_path)

    def parse(self, image_path):
        """Both stages -> {"image","markdown","blocks","layout"}."""
        layout = self.layout.detect(image_path)
        out = self.ocr.run(image_path, layout)
        out["layout"] = layout
        return out
