"""
infer_qwen35_nothink.py
-----------------------
Re-run of CANS/vlm-probing/scripts/infer_qwen35.py with Qwen3.5's thinking mode OFF.

The original run left thinking on (the chat-template default) with max_new_tokens=64,
so every one of the 200 generations per task stopped inside the <think> trace and the
saved attention covers reasoning tokens, not a caption. Here:
  - enable_thinking=False   (template emits an empty <think></think> block in the
                             prompt, so every generated token belongs to the answer)
  - do_sample=False         (greedy; deterministic, so a 3-image run and the full run
                             agree on shared images)
  - max_new_tokens=96       (room for one sentence; generation stops at EOS)
Everything else (prompts, layers, hook, image mask, file layout) matches the original,
so rebuttal/shared/vlm_maps.py-style readers work unchanged on the new directory.

Outputs: CANS/vlm-probing/output/{coco,social_cap}/qwen35_nothink/
           activations/attn-layer-{L}/{img}.pt, image_mask.pt, responses.csv
  --nsd:   rebuttal/task8_nsd/vlm_out/qwen35_nothink_{task}/  (1,098 NSD stimuli; only the
           layers holding the behaviourally selected heads, 7 and 15, to save disk)

Usage:
    python infer_qwen35_nothink.py --task describe [--images 000000157601 ...] [--nsd]
"""
import argparse
import os
import traceback

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['describe', 'social'], required=True)
parser.add_argument('--images', nargs='*', default=None, help='Only these image ids (no .jpg)')
parser.add_argument('--nsd', action='store_true', help='Run on the NSD stimuli instead of the 200 COCO images')
args = parser.parse_args()
TASK = args.task

HF_CACHE   = '/scratch/gpfs/GRAZIANO/hf_cache'
ROOT       = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
MODEL_NAME = 'Qwen/Qwen3.5-9B'
os.environ['HF_HOME'] = HF_CACHE
MAX_NEW_TOKENS = 96

TASK_CFG = {
    'describe': {
        'metadata': os.path.join(ROOT, 'output', 'coco', 'metadata.csv'),
        'prompt':   'Describe the image in one clear, succint sentence. e.g., A sandy beach with colorful umbrellas, people sunbathing, and children playing near the shoreline.',
        'out_dir':  os.path.join(ROOT, 'output', 'coco', 'qwen35_nothink'),
    },
    'social': {
        'metadata': os.path.join(ROOT, 'output', 'social_cap', 'metadata.csv'),
        'prompt':   'Select one person in the image and describe what they are paying attention to. Output one clear, succinct sentence, e.g., The woman in the red dress is paying attention to the cat.',
        'out_dir':  os.path.join(ROOT, 'output', 'social_cap', 'qwen35_nothink'),
    },
}
cfg     = dict(TASK_CFG[TASK])
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
if args.nsd:
    NSD = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd'
    cfg['metadata'] = os.path.join(NSD, 'nsd_metadata.csv')
    cfg['out_dir'] = os.path.join(NSD, 'vlm_out', f'qwen35_nothink_{TASK}')
    FULL_ATTN_LAYERS = [7, 15]
OUT_DIR = cfg['out_dir']
os.makedirs(OUT_DIR, exist_ok=True)

print(f'Task: {TASK}  (thinking OFF, greedy, max_new_tokens={MAX_NEW_TOKENS})')
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, torch_dtype='auto', device_map='auto', attn_implementation='eager',
    local_files_only=True, trust_remote_code=True,
)
model.config.output_attentions = True
processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
device = next(model.parameters()).device
model.eval()

# ── Hooks: keep decode-step attention only (query length 1), as in the original ──
activations: dict[str, list] = {}

def make_attn_hook(layer_name):
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            w = output[1]
            if w is not None and w.size(2) == 1:
                activations.setdefault(layer_name, []).append(w.detach().cpu())
    return hook

hooks = [model.model.language_model.layers[i].self_attn.register_forward_hook(make_attn_hook(f'attn-layer-{i}'))
         for i in FULL_ATTN_LAYERS]

def get_image_mask(input_ids):
    tok = processor.tokenizer
    vs, ve = tok.convert_tokens_to_ids('<|vision_start|>'), tok.convert_tokens_to_ids('<|vision_end|>')
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    s = (input_ids == vs).nonzero(as_tuple=True)[0]; e = (input_ids == ve).nonzero(as_tuple=True)[0]
    if len(s) and len(e):
        mask[s[0]:e[0] + 1] = True
    return mask.reshape(1, -1)

def collect_activations():
    result = {}
    for name, steps in activations.items():
        if not steps:
            continue
        max_seq = max(t.shape[-1] for t in steps)
        padded = [F.pad(t, (0, max_seq - t.shape[-1]), value=float('nan')) for t in steps]
        result[name] = torch.cat(padded, dim=2)
    activations.clear()
    return result

# ── Main loop ─────────────────────────────────────────────────────────────────
metadata = pd.read_csv(cfg['metadata'])
act_base = os.path.join(OUT_DIR, 'activations', f'attn-layer-{FULL_ATTN_LAYERS[0]}')
done_ids = {f[:-3] for f in os.listdir(act_base) if f.endswith('.pt')} if os.path.isdir(act_base) else set()
print(f'Already done: {len(done_ids)} / {len(metadata)}')

responses_path = os.path.join(OUT_DIR, 'responses.csv')
responses_df = (pd.read_csv(responses_path, dtype=str) if os.path.exists(responses_path)
                else pd.DataFrame(columns=['image_name', 'caption', 'n_new_tokens', 'hit_max_tokens']))
mask_path = os.path.join(OUT_DIR, 'image_mask.pt')
mask_saved = os.path.exists(mask_path)

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    img_path = row['path']
    if not os.path.exists(img_path):
        img_path = img_path.replace('/scratch/gpfs/isaacrc/', '/scratch/gpfs/GRAZIANO/')
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    if img_name in done_ids or (args.images and img_name not in args.images):
        continue

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': [{'type': 'image', 'image': f'file://{img_path}'},
                                     {'type': 'text', 'text': cfg['prompt']}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    assert text.rstrip().endswith('</think>'), f'thinking not disabled: ...{text[-80:]!r}'
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors='pt').to(device)

    activations.clear()
    try:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                       output_attentions=True, return_dict_in_generate=True)
    except Exception as e:
        print(f'ERROR on {img_name}: {e}'); traceback.print_exc(); activations.clear(); continue

    new_tok = generated.sequences[0, inputs['input_ids'].shape[1]:]
    caption = processor.tokenizer.decode(new_tok, skip_special_tokens=True).strip()
    n_new = int(new_tok.numel())

    if not mask_saved:
        image_mask = get_image_mask(inputs['input_ids'][0])
        torch.save(image_mask.cpu(), mask_path)
        print(f'\nSaved image_mask: {tuple(image_mask.shape)}, n_image_tokens={image_mask.sum().item()}')
        mask_saved = True

    for name, tensor in collect_activations().items():
        d = os.path.join(OUT_DIR, 'activations', name); os.makedirs(d, exist_ok=True)
        torch.save(tensor, os.path.join(d, f'{img_name}.pt'))

    responses_df = pd.concat([responses_df, pd.DataFrame([{
        'image_name': img_name, 'caption': caption, 'n_new_tokens': n_new,
        'hit_max_tokens': n_new >= MAX_NEW_TOKENS}])], ignore_index=True)
    responses_df.to_csv(responses_path, index=False)
    print(f'{img_name}: [{n_new} tok] {caption}', flush=True)

for h in hooks:
    h.remove()
print(f'\nDone. Responses saved to {responses_path}')
