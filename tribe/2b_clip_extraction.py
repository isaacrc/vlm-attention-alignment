"""
2b_clip_extraction.py
---------------------
Extract CLIP ViT-L/14 visual embeddings for all 200 target images.
Output: tribe/outputs/X_clip.npy  shape (200, 768)

Run on CPU (fast enough — ~200 images in <1 min).
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

BASE       = Path('/scratch/gpfs/GRAZIANO/CANS')
IMG_DIR    = BASE / 'rnn_model/archive_analysis_emmet/analysis/data/val2017'
TARGET_CSV = BASE / 'rnn_model/results/target_imgs.csv'
OUT_PATH   = BASE / 'tribe/outputs/X_clip.npy'

if OUT_PATH.exists():
    print(f'Already exists: {OUT_PATH.name} — skipping.')
    raise SystemExit(0)

target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_imgs = [x.replace('.jpg', '') for x in target_imgs]
N = len(target_imgs)
print(f'Target images: {N}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model     = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model.eval()

BATCH = 16
X = np.zeros((N, 768), dtype=np.float32)

with torch.no_grad():
    for start in tqdm(range(0, N, BATCH), desc='CLIP'):
        batch_ids  = target_imgs[start:start + BATCH]
        imgs       = [Image.open(IMG_DIR / f'{img_id}.jpg').convert('RGB')
                      for img_id in batch_ids]
        inputs     = processor(images=imgs, return_tensors='pt').to(device)
        # Use vision_model + visual_projection directly to avoid API differences
        vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
        pooled     = vision_out.pooler_output                        # (B, 1024)
        feats      = model.visual_projection(pooled)                 # (B, 768)
        feats      = feats / feats.norm(dim=-1, keepdim=True)        # L2-normalize
        X[start:start + len(batch_ids)] = feats.cpu().float().numpy()

np.save(str(OUT_PATH), X)
print(f'Saved: {OUT_PATH.name}  shape={X.shape}')
