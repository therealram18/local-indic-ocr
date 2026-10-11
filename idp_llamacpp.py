"""IndicBlockOCR backend that sends each crop to llama-server over HTTP.

Lets the OCR model run on CPU from quantized GGUF files instead of PyTorch.
Put this file in the repo root, next to indic_ocr.py. Standard library only -- no extra installs.

Start llama-server first:
    llama-server -m indic-ocr-Q8_0.gguf --mmproj mmproj-indic-ocr-f16.gguf -c 8192 --port 8081 -t 6
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request


class LlamaCppRecognizer:
    """Implements the RecognizerBackend protocol: ``transcribe`` returns one string per request."""

    def __init__(self, url: str = "http://127.0.0.1:8081", max_tokens: int = 2048,
                 timeout: float = 900) -> None:
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _one(self, req) -> str:
        buf = io.BytesIO()
        req.image.save(buf, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {
            # Same message shape HfRecognizer builds: image first, then the block's prompt.
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": req.prompt},
            ]}],
            "temperature": 0,  # greedy, like the reference recognizer
            "max_tokens": self.max_tokens,
        }
        http_req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
            content = json.load(resp)["choices"][0]["message"].get("content") or ""
        return content.strip()

    def transcribe(self, requests: list) -> list[str]:
        return [self._one(r) for r in requests]

    def close(self) -> None:
        return None
