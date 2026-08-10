"""IndicDocParser -- page image in, reading-ordered Markdown and per-block JSON out.

    import sys
    from huggingface_hub import snapshot_download

    repo = snapshot_download("bodhan-ai/indic-doc-parser")
    sys.path.insert(0, repo)                       # the code ships in the repo
    from indic_doc_parser import IndicDocParser

    parser = IndicDocParser.from_pretrained(repo)
    print(parser("page.png"))                      # markdown

``sys.path.insert`` is needed because ``snapshot_download`` returns a cache directory, which is
not importable on its own. With the path added, this is ordinary Python -- no
``trust_remote_code``, and nothing of ours to install.

The pipeline is two models, both under ``weights/`` here. Either is usable and finetunable on its
own:

    AutoModelForImageTextToText.from_pretrained(repo, subfolder="weights/ocr")
    AutoModelForObjectDetection.from_pretrained(repo, subfolder="weights/layout",
                                                trust_remote_code=True)

The recognizer is a stock Qwen3.5 checkpoint and needs no custom code at all.
"""

from __future__ import annotations

from pathlib import Path

from idp_offline import IndicBlockOCR as _BlockOCR
from idp_offline import IndicDocLayout as _DocLayout
from idp_types import CropConfig, DedupConfig, LayoutConfig, RecognizerConfig, TableFormat

__all__ = ["IndicDocParser"]


class IndicDocParser:
    """Both stages. Construct with :meth:`from_pretrained`, then call it on a page image."""

    def __init__(
        self,
        path: str | Path,
        device: str = "cuda",
        table_format: str = "html",
        dedup_mode: str = "both",
        contain: float = 0.90,
        min_px_side: int = 256,
        max_new_tokens: int = 2048,
    ) -> None:
        self.path = Path(path)
        self._device = device
        self._layout_cfg = LayoutConfig(device=device)
        self._dedup = DedupConfig(mode=dedup_mode, contain=contain)
        self._crop = CropConfig(min_px_side=min_px_side)
        self._rec_cfg = RecognizerConfig(
            max_tokens=max_new_tokens, table_format=TableFormat(table_format)
        )
        # Both stages are built on first use, so detect() never loads the 1.7 GB recognizer.
        self._layout = None
        self._ocr = None

    @classmethod
    def from_pretrained(cls, path: str | Path | None = None, **kwargs) -> IndicDocParser:
        """Load from a downloaded snapshot. Defaults to the directory this file lives in, which
        is the snapshot itself -- so ``IndicDocParser.from_pretrained()`` also works."""
        return cls(Path(path) if path else Path(__file__).resolve().parent, **kwargs)

    # -- stages ------------------------------------------------------------ #

    @property
    def layout(self):
        if self._layout is None:
            self._layout = _DocLayout(
                ckpt=str(self.path / "weights" / "layout"),
                config=self._layout_cfg,
                dedup=self._dedup,
            )
        return self._layout

    @property
    def recognizer(self):
        if self._ocr is None:
            from idp_recognizer import HfRecognizer

            self._ocr = _BlockOCR(
                backend=HfRecognizer(
                    ckpt=str(self.path / "weights" / "ocr"),
                    config=self._rec_cfg,
                    device=self._device,
                ),
                config=self._rec_cfg,
                dedup=self._dedup,
                crop=self._crop,
            )
        return self._ocr

    # -- public API -------------------------------------------------------- #

    def detect(self, image_path: str) -> dict:
        """Stage 1 only -- blocks, labels, reading order. Loads no recognizer."""
        return self.layout.detect(image_path).as_record()

    def parse(self, image_path: str) -> dict:
        """Both stages -> ``{image, width, height, blocks, markdown}``."""
        page = self.recognizer.run(image_path, self.layout.detect(image_path))
        return {**page.as_record(), "markdown": page.markdown}

    def __call__(self, image_path: str) -> str:
        """The markdown for a page. ``parse()`` if you also want the per-block JSON."""
        return self.parse(image_path)["markdown"]
