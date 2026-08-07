"""Single-image inference for the joint model -> the viewer's content schema.

Produces content=[{bbox:[y0,x0,y1,x1] normalized 0-1000, label, reading_order, score}]
so serve_viewer.py's /infer endpoint can render predictions with the SAME box + reading
-order visualization as the dataset viewer. Decode mirrors src/eval.py exactly (sigmoid-max
detections; reading order = voting-sort over order_logits with the self-diagonal masked).
"""
import numpy as np, torch
from PIL import Image
from .labels import ID2LABEL
def cxcywh_to_xyxy(b):
    c = b.clone()
    c[..., 0], c[..., 1] = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
    c[..., 2], c[..., 3] = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    return c
from .order_loss import pairwise_scores, decode_order

_CACHE = {}


def get_model(ckpt, model_type="ppdoc", device="cuda"):
    """Load (and cache) a trained checkpoint."""
    key = (ckpt, model_type)
    if key not in _CACHE:
        if model_type == "ppdoc":
            from .modeling_ppdoc import PPDocLayoutV3Trainable as M
        else:
            from .modeling import RTDetrV2WithReadingOrder as M
        _CACHE[key] = M.from_pretrained(ckpt).to(device).eval()
    return _CACHE[key]


@torch.no_grad()
def infer(model, pil_img, conf=0.3, img_size=800, device="cuda"):
    im = pil_img.convert("RGB").resize((img_size, img_size))
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255.0)  # [3,S,S] in [0,1]
    out = model(pixel_values=x[None].to(device))
    scores, labels = out.logits.sigmoid().max(-1)                 # [1,N]
    boxes = cxcywh_to_xyxy(out.pred_boxes)[0]                     # [N,4] x0,y0,x1,y1 normalized
    if getattr(out, "order_logits", None) is not None:
        S_full = out.order_logits[0]
    else:
        nq = model.config.num_queries
        S_full = pairwise_scores(out.last_hidden_state[:, -nq:], model.ro_q, model.ro_k)[0]

    keep = (scores[0] > conf).nonzero().squeeze(-1)
    if keep.numel() == 0:
        return []
    kb, ks, kc = boxes[keep], scores[0][keep], labels[0][keep]
    # reading order: correct ppdoc decode over the kept sub-block (triu + (1-S^T).tril)
    S = S_full[keep][:, keep].cpu().float()
    seq = decode_order(S).tolist()                                # reading sequence, first -> last
    rank = [0] * len(seq)
    for r, pos in enumerate(seq):
        rank[pos] = r + 1

    content = []
    for j in range(keep.numel()):
        x0, y0, x1, y1 = kb[j].tolist()
        content.append({
            "bbox": [round(y0 * 1000, 1), round(x0 * 1000, 1),   # viewer wants [y0,x0,y1,x1]
                     round(y1 * 1000, 1), round(x1 * 1000, 1)],
            "label": ID2LABEL[int(kc[j])],
            "reading_order": rank[j],
            "score": round(float(ks[j]), 3),
        })
    return content
