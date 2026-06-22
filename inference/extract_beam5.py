"""
extract_beam5.py
----------------
Extract beam-search attention maps from all checkpoints in a given
condition × phase directory.

Usage:
  python extract_beam5.py --condition ViT_LSTM --phase social_unfrozen
  python extract_beam5.py --condition CNN_Transformer --phase social_frozen

Saves per-checkpoint NPZ to:
  {COND_DIR}/ckpt_cache/{phase}_beam5_E{epoch:03d}.npz

Each NPZ contains:
  alphas       : dict {img_name: (196,) float32}  — last-layer, mean over words
  layer_alphas : dict {img_name: (num_layers, 196) float32} — all layers, mean over words
  caption      : dict {img_name: str}
"""
import os, sys, re, json, argparse
import numpy as np
import torch
import pandas as pd
from imageio import imread
from skimage.transform import resize
import torchvision.transforms as transforms
from tqdm import tqdm

ARCH_DIR = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
RNN_DIR  = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
IMG_DIR  = os.path.join(RNN_DIR, 'data', 'coco_imgs', 'val2014')
TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
WORD_MAP_PATH = os.path.join(RNN_DIR, 'data', 'splits',
                             'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')

sys.path.insert(0, RNN_DIR)
sys.path.insert(0, os.path.join(ARCH_DIR, 'shared'))
sys.path.insert(0, os.path.join(ARCH_DIR, 'models'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORMER_CONDITIONS = {'CNN_Transformer', 'ViT_Transformer'}
VIT_CONDITIONS = {'ViT_LSTM', 'ViT_Transformer'}  # require 224x224 input


def load_image(image_path, resize_to=256):
    img = imread(image_path)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    if img.shape[-1] > 3:
        img = img[..., :3]
    img = resize(img, (resize_to, resize_to))
    img = img.transpose(2, 0, 1).astype(np.float32)
    t = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize(t).unsqueeze(0)


def extract_lstm(encoder, decoder, img_path, word_map, beam_size=5, img_size=256):
    """Run beam search for LSTM decoder; returns (seq, alphas_196, layer_alphas, None)."""
    from final_caption_2 import caption_image_beam_search
    seq, alphas, emb, h, c = caption_image_beam_search(
        encoder, decoder, img_path, word_map, False, 0.5, beam_size, resize_to=img_size)
    alphas = torch.as_tensor(alphas).detach().cpu()  # (T, 14, 14)
    mean_alpha = alphas.mean(dim=0).numpy().flatten()  # (196,)
    # LSTM has single attention layer; no per-head data
    layer_alphas = mean_alpha[np.newaxis, :]           # (1, 196)
    return seq, mean_alpha, layer_alphas, None


def extract_transformer(encoder, decoder, img_path, word_map, beam_size=5, img_size=256):
    """Run beam search for Transformer decoder; returns (seq, alphas_196, layer_alphas, layer_alphas_perhead)."""
    from caption_transformer import caption_image_beam_search_transformer
    seq, alphas, layer_alphas, layer_alphas_perhead = caption_image_beam_search_transformer(
        encoder, decoder, img_path, word_map, beam_size, resize_to=img_size)
    # alphas: (T, 196), layer_alphas: (num_layers, T, 196), layer_alphas_perhead: (num_layers, nhead, T, 196)
    mean_alpha        = alphas.mean(axis=0)              # (196,)
    mean_layer_alpha  = layer_alphas.mean(axis=1)        # (num_layers, 196)
    mean_layer_ph     = layer_alphas_perhead.mean(axis=2)  # (num_layers, nhead, 196)
    return (seq,
            mean_alpha.astype(np.float32),
            mean_layer_alpha.astype(np.float32),
            mean_layer_ph.astype(np.float32))


def epoch_from_path(path):
    m = re.search(r'epoch(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                        choices=['CNN_Transformer', 'ViT_LSTM', 'ViT_Transformer'])
    parser.add_argument('--phase', required=True,
                        choices=['phase1', 'phase2', 'social_unfrozen', 'social_frozen'])
    parser.add_argument('--beam_size', type=int, default=5)
    args = parser.parse_args()

    cond_dir   = os.path.join(ARCH_DIR, args.condition)
    ckpt_dir   = os.path.join(cond_dir, 'checkpoints', args.phase)
    cache_dir  = os.path.join(cond_dir, 'ckpt_cache')
    os.makedirs(cache_dir, exist_ok=True)

    is_transformer = args.condition in TRANSFORMER_CONDITIONS
    img_size = 224 if args.condition in VIT_CONDITIONS else 256

    target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
    print(f'Target images: {len(target_imgs)}')

    with open(WORD_MAP_PATH) as f:
        word_map = json.load(f)
    rev_word_map = {v: k for k, v in word_map.items()}

    ckpts = sorted([
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')
    ])
    print(f'Found {len(ckpts)} checkpoints in {ckpt_dir}')

    for ckpt_path in ckpts:
        epoch = epoch_from_path(ckpt_path)
        out_path = os.path.join(cache_dir,
                                f'{args.phase}_beam5_E{epoch:03d}.npz')
        if os.path.exists(out_path):
            print(f'  Skipping (cached): {out_path}')
            continue

        print(f'\nLoading checkpoint: {os.path.basename(ckpt_path)}')
        ckpt    = torch.load(ckpt_path, map_location='cpu')
        encoder = ckpt['encoder'].to(device).eval()
        decoder = ckpt['decoder'].to(device).eval()

        results = {'alphas': {}, 'layer_alphas': {}, 'head_layer_alphas': {}, 'caption': {}}

        for img_name in tqdm(target_imgs, desc=f'  E{epoch:03d}'):
            img_path = os.path.join(IMG_DIR, 'COCO_val2014_' + img_name)
            if not os.path.exists(img_path):
                print(f'  WARNING: missing {img_path}')
                continue
            try:
                if is_transformer:
                    seq, alpha, layer_alpha, layer_alpha_ph = extract_transformer(
                        encoder, decoder, img_path, word_map, args.beam_size, img_size)
                else:
                    seq, alpha, layer_alpha, layer_alpha_ph = extract_lstm(
                        encoder, decoder, img_path, word_map, args.beam_size, img_size)
            except Exception as e:
                print(f'  ERROR on {img_name}: {e}')
                continue

            words = [rev_word_map[i] for i in seq
                     if i not in (word_map['<start>'], word_map['<end>'],
                                  word_map.get('<pad>', -1))]
            results['alphas'][img_name]       = alpha.astype(np.float32)
            results['layer_alphas'][img_name] = layer_alpha.astype(np.float32)
            if layer_alpha_ph is not None:
                results['head_layer_alphas'][img_name] = layer_alpha_ph.astype(np.float32)
            results['caption'][img_name]      = ' '.join(words)

        np.savez(out_path, **results)
        print(f'  Saved: {out_path}')

        # Free GPU memory between checkpoints
        del encoder, decoder
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
