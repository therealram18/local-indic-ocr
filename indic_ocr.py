"""IndicOCR -- page image in, reading-ordered Markdown and per-block JSON out.

    import sys
    from huggingface_hub import snapshot_download

    repo = snapshot_download("bodhan-ai/indic-ocr")
    sys.path.insert(0, repo)                       # the code ships in the repo
    from indic_ocr import IndicOCR

    parser = IndicOCR.from_pretrained(repo)
    print(parser("page.png"))                      # markdown

``sys.path.insert`` is needed because ``snapshot_download`` returns a cache directory, which is
not importable on its own. With the path added, this is ordinary Python -- no
``trust_remote_code``, and nothing of ours to install.

The two stages are also public, and each runs on its own:

    IndicDocLayout(f"{repo}/weights/layout").detect("page.png")   # blocks and reading order
    IndicBlockOCR(f"{repo}/weights/ocr").run("page.png", layout)  # layout may be your own

``layout`` there can be a PageResult, a dict, or the path to a layout JSON file, so the stages
compose across processes and you can correct a layout before transcribing it.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def _preflight() -> None:
    """One clear error instead of a cascade of ImportErrors from inside transformers.

    Every failure here is one someone has actually hit: a torch built for a CUDA line the driver
    cannot run (silently CPU), a torchvision that does not match its torch, or a transformers
    older than PPDocLayoutV3.
    """
    fix = "./install.sh  (or see the Installation section of the model card)"
    for name, floor in (("torch", (2, 4)), ("transformers", (5, 7))):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            raise ImportError(f"IndicOCR needs {name}. Run: {fix}") from None
        got = tuple(int(p) for p in mod.__version__.split(".")[:2] if p.isdigit())
        if got < floor:
            raise ImportError(
                f"IndicOCR needs {name}>={'.'.join(map(str, floor))}, "
                f"found {mod.__version__}. Run: {fix}"
            )
    # torchvision is stage 2 only, so its absence is not fatal here -- the layout stage runs
    # without it. It is checked where the recognizer is built.


_preflight()

# ruff: noqa: E402  -- the preflight has to run before transformers is imported, which is the
# whole point of it; these imports pull transformers in transitively.
from idp_offline import IndicBlockOCR, IndicDocLayout
from idp_types import CropConfig, DedupConfig, LayoutConfig, RecognizerConfig, TableFormat

#: The two stages are public: either runs on its own, and IndicBlockOCR accepts a layout you
#: produced or corrected yourself.
__all__ = ["IndicBlockOCR", "IndicDocLayout", "IndicOCR"]


class IndicOCR:
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
    def from_pretrained(cls, path: str | Path | None = None, **kwargs) -> IndicOCR:
        """Load from a downloaded snapshot. Defaults to the directory this file lives in, which
        is the snapshot itself -- so ``IndicOCR.from_pretrained()`` also works."""
        return cls(Path(path) if path else Path(__file__).resolve().parent, **kwargs)

    # -- stages ------------------------------------------------------------ #

    @property
    def layout(self):
        if self._layout is None:
            self._layout = IndicDocLayout(
                ckpt=str(self.path / "weights" / "layout"),
                config=self._layout_cfg,
                dedup=self._dedup,
            )
        return self._layout

    @property
    def recognizer(self):
        if self._ocr is None:
            if importlib.util.find_spec("torchvision") is None:
                raise ImportError(
                    "The recognizer needs torchvision (its image processor uses it); the layout "
                    "stage does not, so detect() still works. Install it together with torch, "
                    "from the same index -- torchvision pins an exact torch version. See "
                    "./install.sh"
                )
            from idp_recognizer import HfRecognizer

            self._ocr = IndicBlockOCR(
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
