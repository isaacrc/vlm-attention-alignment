"""
infer_qwen35.py
---------------
Run Qwen3.5-9B on all 200 COCO images for one task (describe or social),
saving per-layer attention activations (full-attention layers only) and
one shared image_mask.pt.

Full-attention layers (out of 32 total): [3, 7, 11, 15, 19, 23, 27, 31]
Module path: model.model.language_model.layers[i].self_attn

Usage:
    python scripts/infer_qwen35.py --task describe
    python scripts/infer_qwen35.py --task social
"""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['describe', 'social'], required=True)
parser.add_argument('--n_images', type=int, default=None,
                    help='Limit to first N images (for quick CLI testing)')
args = parser.parse_args()

TASK = args.task

# ── Paths ─────────────────────────────────────────────────────────────────────
HF_CACHE   = '/scratch/gpfs/GRAZIANO/hf_cache'
ROOT       = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
MODEL_NAME = 'Qwen/Qwen3.5-9B'
os.environ['HF_HOME'] = HF_CACHE

TASK_CFG = {
    'describe': {
        'metadata':  '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/nsd_metadata.csv',
        'prompt':    'Describe the image in one clear, succint sentence. e.g., A sandy beach with colorful umbrellas, people sunbathing, and children playing near the shoreline.',
        'out_dir':   '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/vlm_out/qwen35_describe',
    },
    'social': {
        'metadata':  '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/nsd_metadata.csv',
        'prompt':    'Select one person in the image and describe what they are paying attention to. Output one clear, succinct sentence, e.g., The woman in the red dress is paying attention to the cat.',
        'out_dir':   '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/vlm_out/qwen35_social',
    },
}

cfg     = TASK_CFG[TASK]
OUT_DIR = cfg['out_dir']
os.makedirs(OUT_DIR, exist_ok=True)

# Qwen3.5-9B: only 8 of 32 layers use full (standard) attention
FULL_ATTN_LAYERS = [15]   # NSD run: only the analysed layer, saves disk
N_HEADS = 16   # num_attention_heads in text_config

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Task: {TASK}")
print("Loading Qwen3.5-9B ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype='auto',
    device_map='auto',
    attn_implementation='eager',
    local_files_only=True,
    trust_remote_code=True,
)
model.config.output_attentions = True
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, local_files_only=True, trust_remote_code=True
)
device = next(model.parameters()).device
model.eval()
print(f"Model loaded on {device}")
print(f"Full-attention layers: {FULL_ATTN_LAYERS}")

# ── Hook infrastructure ───────────────────────────────────────────────────────
activations: dict[str, list] = {}

def make_attn_hook(layer_name):
    def hook(module, input, output):
        try:
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None and attn_weights.size(2) == 1:
                    activations.setdefault(layer_name, []).append(
                        attn_weights.detach().cpu()
                    )
        except Exception as e:
            print(f'Hook error {layer_name}: {e}')
    return hook

hooks = []
for i in FULL_ATTN_LAYERS:
    h = model.model.language_model.layers[i].self_attn.register_forward_hook(
        make_attn_hook(f'attn-layer-{i}')
    )
    hooks.append(h)

# ── Image token mask helper ───────────────────────────────────────────────────
def get_image_mask(input_ids):
    tokenizer = processor.tokenizer
    vid_start = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vid_end   = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    starts = (input_ids == vid_start).nonzero(as_tuple=True)[0]
    ends   = (input_ids == vid_end).nonzero(as_tuple=True)[0]
    if len(starts) and len(ends):
        mask[starts[0]:ends[0] + 1] = True
    return mask.reshape(1, -1)

# ── Pad & stack activations (one image) ──────────────────────────────────────
def collect_activations():
    result = {}
    for layer_name, steps in activations.items():
        if not steps:
            continue
        max_seq = max(t.shape[-1] for t in steps)
        padded  = [F.pad(t, (0, max_seq - t.shape[-1]), value=float('nan')) for t in steps]
        stacked = torch.cat(padded, dim=2)
        result[layer_name] = stacked
    activations.clear()
    return result

# ── Main loop ─────────────────────────────────────────────────────────────────
metadata = pd.read_csv(cfg['metadata'])
if args.n_images is not None:
    metadata = metadata.head(args.n_images)
    print(f'Testing with first {args.n_images} images')
prompt = cfg['prompt']

act_base = os.path.join(OUT_DIR, 'activations', f'attn-layer-{FULL_ATTN_LAYERS[0]}')
done_ids = set()
if os.path.isdir(act_base):
    done_ids = {f.replace('.pt', '') for f in os.listdir(act_base) if f.endswith('.pt')}
print(f"Already done: {len(done_ids)} / {len(metadata)}")

responses_path = os.path.join(OUT_DIR, 'responses.csv')
if os.path.exists(responses_path):
    responses_df = pd.read_csv(responses_path)
else:
    responses_df = pd.DataFrame(columns=['image_name', 'caption'])

mask_path = os.path.join(OUT_DIR, 'image_mask.pt')
mask_saved = os.path.exists(mask_path)

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    img_path = row['path']
    if not os.path.exists(img_path):
        img_path = img_path.replace('/scratch/gpfs/isaacrc/', '/scratch/gpfs/GRAZIANO/')
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    if img_name in done_ids:
        continue

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': f'file://{img_path}'},
                {'type': 'text',  'text':  prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    ).to(device)

    activations.clear()
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                output_attentions=True,
                return_dict_in_generate=True,
            )
    except Exception as e:
        print(f'ERROR on {img_name}: {e}')
        traceback.print_exc()
        activations.clear()
        continue

    decoded = processor.batch_decode(generated.sequences, skip_special_tokens=True)
    caption = decoded[0]
    if 'assistant' in caption.lower():
        caption = caption.split('assistant')[-1].strip()

    if not mask_saved:
        image_mask = get_image_mask(inputs['input_ids'][0])
        n_img_tokens = image_mask.sum().item()
        torch.save(image_mask.cpu(), mask_path)
        print(f"\nSaved image_mask: {image_mask.shape}, n_image_tokens={n_img_tokens}")
        mask_saved = True

    act_dict = collect_activations()
    for layer_name, tensor in act_dict.items():
        layer_dir = os.path.join(OUT_DIR, 'activations', layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        torch.save(tensor, os.path.join(layer_dir, f'{img_name}.pt'))

    responses_df = pd.concat(
        [responses_df, pd.DataFrame([{'image_name': img_name, 'caption': caption}])],
        ignore_index=True,
    )
    responses_df.to_csv(responses_path, index=False)

print(f"\nDone. Responses saved to {responses_path}")

for h in hooks:
    h.remove()
