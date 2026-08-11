# IndicDocParser: troubleshooting

Install and runtime problems, and what to do about them. The model card is
[README.md](README.md).

Nearly every install problem is the torch/torchvision/CUDA triangle.

**Installing without the script**, on Colab, Windows, or anywhere a shell script is awkward. These
are the two commands `install.sh` runs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # your CUDA line
pip install -r requirements.txt
```

Keep `torch` and `torchvision` in that one command. `torchvision` pins an exact `torch`, so
installing them separately lets pip pick a mismatched pair. A plain `pip install torch` also takes
the newest wheel, currently CUDA 13, which on an older driver initialises to CPU with nothing but a
warning. Use the index for *your* CUDA: `cu121`, `cu124`, `cu128`, `cu130`, or `cpu`.

**`ImportError` after `ImportError`, or complaints that torchvision is too old.** `torchvision`
pins an exact `torch`, so if the two were installed separately pip may have paired them wrongly.
Reinstall both in one command, from one index:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**`CUDA initialization: The NVIDIA driver on your system is too old`, then everything runs on CPU.**
Plain `pip install torch` fetched a wheel built for a newer CUDA than your driver. Check with
`nvidia-smi`, then reinstall from the matching index. Verify with:

```python
import torch; print(torch.__version__, torch.cuda.is_available())
```

**`Qwen3VLVideoProcessor requires the Torchvision library`.** The recognizer needs `torchvision`;
the layout stage does not. Install it *together with* torch, from the same index.

**`RTDetrHungarianMatcher requires the scipy library`.** You are on an old revision of this repo.
Update to the current revision, which no longer builds the matcher during inference.
