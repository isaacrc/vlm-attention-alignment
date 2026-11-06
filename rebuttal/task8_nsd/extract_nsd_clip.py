"""CLIP ViT-L/14 visual embeddings for the 1098 NSD stimuli — the "what" band of the
banded ridge, matching tribe/scripts/2b_clip_extraction.py.

Output: features/X_clip_nsd.npy  (1098, 768), rows ordered by stimuli_index.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

HERE = Path(__file__).parent
OUT = HERE / "features"
OUT.mkdir(exist_ok=True)
OUT_PATH = OUT / "X_clip_nsd.npy"

if OUT_PATH.exists():
    print(f"exists: {OUT_PATH}")
    raise SystemExit(0)

idx = pd.read_csv(HERE / "stimuli_index.csv")
N = len(idx)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{N} images on {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

BATCH = 16
X = np.zeros((N, 768), dtype=np.float32)
with torch.no_grad():
    for s in tqdm(range(0, N, BATCH), desc="CLIP"):
        ids = idx["nsdId"].iloc[s:s + BATCH]
        imgs = [Image.open(HERE / "stimuli" / f"{int(i):05d}.png").convert("RGB") for i in ids]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        X[s:s + len(imgs)] = model.get_image_features(**inp).cpu().numpy()

np.save(OUT_PATH, X)
print(f"saved {OUT_PATH} {X.shape}")
