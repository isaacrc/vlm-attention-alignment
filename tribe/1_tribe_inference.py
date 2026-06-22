"""
1_tribe_inference.py
--------------------
For each of the 200 COCO target images, create a 30-second static MP4
and run TRIBE (video-only) to get predicted fMRI on fsaverage5 (~20484 vertices).
Average over TRs to produce one brain vector per image.

Output: tribe/outputs/tribe_Y.npy  shape (200, 20484)

Run on GPU node via tribe/slurm/tribe_inference.sh
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE      = Path('/scratch/gpfs/GRAZIANO/CANS')
TRIBE_DIR = BASE / 'tribe'
IMG_DIR   = BASE / 'rnn_model/archive_analysis_emmet/analysis/data/val2017'
TARGET_CSV = BASE / 'rnn_model/results/target_imgs.csv'
OUT_PATH  = TRIBE_DIR / 'outputs/tribe_Y.npy'
CACHE_DIR = TRIBE_DIR / 'cache'
VIDEO_DIR = TRIBE_DIR / 'outputs/videos'

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print('Loading TribeModel...')
sys.path.insert(0, str(TRIBE_DIR))
from tribev2 import TribeModel
model = TribeModel.from_pretrained(
    str(TRIBE_DIR / 'model_weights'),
    cache_folder=str(CACHE_DIR),
)
print('Model loaded.')

# ── Target images ──────────────────────────────────────────────────────────────
target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_imgs = [x.replace('.jpg', '') for x in target_imgs]
print(f'Target images: {len(target_imgs)}')

# ── Inference loop ─────────────────────────────────────────────────────────────
N_VERTICES = 20484

# Resume from checkpoint if it exists
if OUT_PATH.exists():
    all_preds = np.load(str(OUT_PATH))
    done_mask = all_preds.any(axis=1)   # rows with any non-zero = already processed
    n_done = int(done_mask.sum())
    print(f'Resuming from checkpoint: {n_done}/{len(target_imgs)} images already done')
else:
    all_preds = np.zeros((len(target_imgs), N_VERTICES), dtype=np.float32)
    done_mask = np.zeros(len(target_imgs), dtype=bool)
    n_done = 0

for i, img_id in enumerate(tqdm(target_imgs, desc='TRIBE inference')):
    if done_mask[i]:
        continue   # already processed in a previous run
    img_path   = IMG_DIR / f'{img_id}.jpg'
    video_path = VIDEO_DIR / f'{img_id}.mp4'

    # Create 30-second static video (cached — skip if already exists)
    if not video_path.exists():
        from moviepy import ImageClip
        clip = ImageClip(str(img_path), duration=30)
        clip.write_videofile(
            str(video_path), codec='libx264', audio=False, fps=10, logger=None
        )

    # Build Video-only events manually — bypasses audio extraction and
    # transcription (whisperx/w2v-bert/llama), so only vjepa2 is needed.
    from neuralset.events.utils import standardize_events
    from neuralset.events.transforms import ChunkEvents
    raw = pd.DataFrame([{
        "type": "Video", "filepath": str(video_path),
        "start": 0, "timeline": "default", "subject": "default",
    }])
    events = ChunkEvents(
        event_type_to_chunk="Video", max_duration=60, min_duration=30
    )(standardize_events(raw))
    events = standardize_events(events)

    preds, _ = model.predict(events, verbose=False)   # (n_TRs, 20484)
    all_preds[i] = preds.mean(axis=0)

    if (i + 1) % 10 == 0:
        np.save(str(OUT_PATH), all_preds)  # checkpoint every 10 images
        print(f'  Saved checkpoint at image {i+1}')

np.save(str(OUT_PATH), all_preds)
print(f'Saved: {OUT_PATH}  shape={all_preds.shape}')
