import sys
from pathlib import Path
import argparse
# from huggingface_hub import snapshot_download

# repo = snapshot_download("bodhan-ai/indic-ocr")
# sys.path.insert(0, indic_ocr)                    # the code ships in the repo
# sys.path.insert(0, Path.cwd().parent)  # the code ships in the repo
from indic_ocr import IndicOCR

repo = sys.path[0]
parser = IndicOCR.from_pretrained(repo, table_format="markdown")

if __name__ == "__main__":

  prs = argparse.ArgumentParser(description="IndicOCR Custom Inference")
  prs.add_argument("--image", type=str, help="Path to the input image")
  args = prs.parse_args()

  img_path = Path(args.image)
  if not img_path.is_file():
        raise ValueError("Invalid image path. Please provide a valid path to the input image.")

  page = parser.parse(img_path)
  print(page["markdown"])                     # reading-ordered Markdown
