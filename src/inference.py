import sys
from pathlib import Path
import argparse

from idp_types import LayoutConfig, CropConfig, RecognizerConfig, TableFormat
from idp_offline import IndicBlockOCR, IndicDocLayout
from idp_llamaccp import LlamaCppRecognizer
# from huggingface_hub import snapshot_download

# repo = snapshot_download("bodhan-ai/indic-ocr")
# sys.path.insert(0, indic_ocr)                    # the code ships in the repo
# sys.path.insert(0, Path.cwd().parent)  # the code ships in the repo
# from indic_ocr import IndicOCR

repo = Path(sys.path[0])

# Layout model
layout = IndicDocLayout(str(repo/"weights"/"layout")),
LayoutConfig(device='cpu')

# OCR model: served by llama-server
ocr = IndicBlockOCR(
    config=RecognizerConfig(table_format=TableFormat("markdown")),
    crop=CropConfig(min_px_side=256, max_px_side=1024),
    backend=LlamaCppRecognizer("http://127.0.0.1:8081"),
)
parser = IndicOCR.from_pretrained(repo, table_format="markdown")
parser._rec_cfg = parser._rec_cfg.merged(dtype="int8")
# parser._layout_cfg = parser._layout_cfg.merged(dtype="int8")
parser._layout_cfg = LayoutConfig(device='cpu')
parser.recognizer.crop = CropConfig(min_px_side=256, max_px_side=1024)

if __name__ == "__main__":

  prs = argparse.ArgumentParser(description="IndicOCR Custom Inference")
  prs.add_argument("--image", type=str, help="Path to the input image")
  args = prs.parse_args()

  img_path = Path(args.image)
  if not img_path.is_file():
        raise ValueError("Invalid image path. Please provide a valid path to the input image.")

  page = parser.parse(img_path)
  print(page["markdown"])                     # reading-ordered Markdown
