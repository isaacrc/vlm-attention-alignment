"""
run_neglect.py
--------------
Hemispatial neglect ablation sweep for all 4 architecture conditions.

Usage:
  python run_neglect.py --condition CNN_LSTM       --task social --layer all
  python run_neglect.py --condition ViT_LSTM       --task describe --layer all
  python run_neglect.py --condition CNN_Transformer --task social --layer 2
  python run_neglect.py --condition ViT_Transformer --task social --layer all

--layer: 'all' (mask all decoder layers) or '0','1','2','3' (single layer).
         For LSTM conditions --layer is ignored (single attention layer).

Checkpoint selection (best spatial_M from summary_table.csv):
  CNN_LSTM    social:   rnn_model/current_models/BEST_checkpoint_CANS_1_..._final.pth.tar
  CNN_LSTM    describe: rnn_model/current_models/BEST_checkpoint_coco_..._final.pth.tar
  CNN_Transformer social:   social_unfrozen/epoch002_bleu0.1376.pth.tar
  CNN_Transformer describe: phase2/epoch000_bleu0.1863.pth.tar
  ViT_LSTM    social:   social_unfrozen/epoch000_bleu0.1476.pth.tar
  ViT_LSTM    describe: phase2/epoch002_bleu0.2411.pth.tar
  ViT_Transformer social:   social_unfrozen/epoch000_bleu0.1203.pth.tar
  ViT_Transformer describe: phase2/epoch002_bleu0.1931.pth.tar

Output:
  {ARCH_DIR}/{condition}/neglect/{task}_neglect_{layer_str}.npy
  Shape: (200, 15) — index 0 = unablated baseline (1.0), indices 1-14 = 14 ablation levels
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import pandas as pd
from tqdm import tqdm

# ── SentenceTransformer must be imported and initialized BEFORE sys.path is
#    modified — rnn_model/datasets.py would otherwise shadow the HuggingFace
#    datasets package that sentence_transformers lazily imports at runtime.
from sentence_transformers import SentenceTransformer
_EMB_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ── paths ──────────────────────────────────────────────────────────────────────
ARCH_DIR     = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
RNN_DIR      = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
IMG_DIR      = os.path.join(RNN_DIR, 'data', 'coco_imgs', 'val2014')
TARGET_CSV   = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
WORD_MAP_PATH = os.path.join(RNN_DIR, 'data', 'splits',
                              'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')

sys.path.insert(0, RNN_DIR)
sys.path.insert(0, os.path.join(ARCH_DIR, 'shared'))
sys.path.insert(0, os.path.join(ARCH_DIR, 'models'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── checkpoint map ─────────────────────────────────────────────────────────────
CHECKPOINTS = {
    ('CNN_LSTM', 'social'):
        os.path.join(RNN_DIR, 'current_models',
                     'BEST_checkpoint_CANS_1_cap_per_img_5_min_word_freq_final.pth.tar'),
    ('CNN_LSTM', 'describe'):
        os.path.join(RNN_DIR, 'current_models',
                     'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq_final.pth.tar'),
    ('CNN_Transformer', 'social'):
        os.path.join(ARCH_DIR, 'CNN_Transformer', 'checkpoints',
                     'social_unfrozen', 'epoch002_bleu0.1376.pth.tar'),
    ('CNN_Transformer', 'describe'):
        os.path.join(ARCH_DIR, 'CNN_Transformer', 'checkpoints',
                     'phase2', 'epoch000_bleu0.1863.pth.tar'),
    ('ViT_LSTM', 'social'):
        os.path.join(ARCH_DIR, 'ViT_LSTM', 'checkpoints',
                     'social_unfrozen', 'epoch000_bleu0.1476.pth.tar'),
    ('ViT_LSTM', 'describe'):
        os.path.join(ARCH_DIR, 'ViT_LSTM', 'checkpoints',
                     'phase2', 'epoch002_bleu0.2411.pth.tar'),
    ('ViT_Transformer', 'social'):
        os.path.join(ARCH_DIR, 'ViT_Transformer', 'checkpoints',
                     'social_unfrozen', 'epoch000_bleu0.1203.pth.tar'),
    ('ViT_Transformer', 'describe'):
        os.path.join(ARCH_DIR, 'ViT_Transformer', 'checkpoints',
                     'phase2', 'epoch002_bleu0.1931.pth.tar'),
}

TRANSFORMER_CONDITIONS = {'CNN_Transformer', 'ViT_Transformer'}
VIT_CONDITIONS         = {'ViT_LSTM', 'ViT_Transformer'}

# 14 ablation levels: remove 1 column at a time from left (1/14 → 14/14)
ABLATION_LEVELS = [i / 14 for i in range(1, 15)]


# ── helpers ────────────────────────────────────────────────────────────────────

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


def get_sentence_emb(seq, rev_word_map, emb_model):
    words    = [rev_word_map[i] for i in seq
                if rev_word_map.get(i, '') not in ('<start>', '<end>', '<pad>')]
    sentence = ' '.join(words)
    return emb_model.encode(sentence)


def load_model(condition, task):
    ckpt_path = CHECKPOINTS[(condition, task)]
    print(f'  Loading: {os.path.basename(ckpt_path)}')
    ckpt    = torch.load(ckpt_path, map_location='cpu')
    encoder = ckpt['encoder'].to(device).eval()
    decoder = ckpt['decoder'].to(device).eval()
    return encoder, decoder


def caption_lstm(encoder, decoder, img_path, word_map, add_noise, noise_pct, img_size):
    from final_caption_2 import caption_image_beam_search
    seq, alphas, emb, h, c = caption_image_beam_search(
        encoder, decoder, img_path, word_map,
        add_noise, noise_pct, beam_size=5, resize_to=img_size)
    return seq


def caption_transformer(encoder, decoder, img_path, word_map,
                        ablation_pct, target_layer, img_size):
    from caption_transformer import caption_image_beam_search_transformer
    seq, alphas, layer_alphas, _ = caption_image_beam_search_transformer(
        encoder, decoder, img_path, word_map,
        beam_size=5, resize_to=img_size,
        ablation_pct=ablation_pct, target_layer=target_layer)
    return seq


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                        choices=['CNN_LSTM', 'CNN_Transformer', 'ViT_LSTM', 'ViT_Transformer'])
    parser.add_argument('--task', required=True, choices=['social', 'describe'])
    parser.add_argument('--layer', default='all',
                        help="'all' or single layer index 0-3 (ignored for LSTM conditions)")
    args = parser.parse_args()

    condition = args.condition
    task      = args.task
    is_transformer = condition in TRANSFORMER_CONDITIONS
    img_size  = 224 if condition in VIT_CONDITIONS else 256

    # Parse --layer
    if args.layer == 'all':
        target_layer = None
        layer_str    = 'all_layers'
    else:
        target_layer = int(args.layer)
        layer_str    = f'layer{target_layer}'
        if not is_transformer:
            print(f'WARNING: --layer {args.layer} ignored for LSTM condition {condition}.')
            layer_str = 'all_layers'

    out_dir  = os.path.join(ARCH_DIR, condition, 'neglect')
    out_path = os.path.join(out_dir, f'{task}_neglect_{layer_str}.npy')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f'Output already exists, skipping: {out_path}')
        return

    print(f'\n=== {condition} | task={task} | layer={layer_str} ===')
    print(f'  device: {device}')

    with open(WORD_MAP_PATH) as f:
        word_map = json.load(f)
    rev_word_map = {v: k for k, v in word_map.items()}

    target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
    print(f'  Images: {len(target_imgs)}')

    emb_model = _EMB_MODEL
    encoder, decoder = load_model(condition, task)

    cos_sims = []

    for img_name in tqdm(target_imgs, desc=f'{condition}/{task}/{layer_str}'):
        img_path = os.path.join(IMG_DIR, 'COCO_val2014_' + img_name)
        if not os.path.exists(img_path):
            print(f'  WARNING: missing {img_path}')
            cos_sims.append([1.0] + [np.nan] * len(ABLATION_LEVELS))
            continue

        # Unablated baseline caption
        try:
            if is_transformer:
                seq_base = caption_transformer(encoder, decoder, img_path, word_map,
                                               0.0, None, img_size)
            else:
                seq_base = caption_lstm(encoder, decoder, img_path, word_map,
                                        False, 0.0, img_size)
        except Exception as e:
            print(f'  ERROR baseline {img_name}: {e}')
            cos_sims.append([1.0] + [np.nan] * len(ABLATION_LEVELS))
            continue

        emb_base = get_sentence_emb(seq_base, rev_word_map, emb_model)
        sims     = [1.0]  # baseline similarity with itself

        # Ablation sweep
        for ablation_pct in ABLATION_LEVELS:
            try:
                if is_transformer:
                    seq_abl = caption_transformer(encoder, decoder, img_path, word_map,
                                                  ablation_pct, target_layer, img_size)
                else:
                    seq_abl = caption_lstm(encoder, decoder, img_path, word_map,
                                           True, ablation_pct, img_size)
            except Exception as e:
                print(f'  ERROR ablation {img_name} pct={ablation_pct:.2f}: {e}')
                sims.append(np.nan)
                continue

            emb_abl = get_sentence_emb(seq_abl, rev_word_map, emb_model)
            sims.append(cosine_similarity(emb_base, emb_abl))

        cos_sims.append(sims)

    arr = np.array(cos_sims, dtype=np.float32)  # (200, 15)
    np.save(out_path, arr)
    print(f'\nSaved: {out_path}  shape={arr.shape}')


if __name__ == '__main__':
    main()
