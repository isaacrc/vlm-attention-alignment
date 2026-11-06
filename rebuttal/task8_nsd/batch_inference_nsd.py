from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import sys
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


# Define your model and image location here
IMAGE_PATH = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/data/000000463618.jpg'
MAX_NEW_TOKENS = 32
MODEL_DIR = "/scratch/gpfs/GRAZIANO/CANS/molmo/molmo/Molmo-7B-D-0924-local/models--allenai--Molmo-7B-D-0924/snapshots/ac032b93b84a7f10c9578ec59f9f20ee9a8990a2"
META_DATA_PATH = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd'

# dynamic vars ## 
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--task', choices=['describe','social'], required=True)
_args = _ap.parse_args()
TASK = _args.task
#PROMPT = "Count the number of people in this image"
PROMPT = {
  "social": "Select one person in the image and describe what they are paying attention to. Output one clear, succinct sentence, e.g., 'The woman in the red dress is paying attention to the cat.'",
  "describe": "Describe the image in one clear, succint sentence. Limit verbosity.",
}[TASK]
#PROMPT = "Describe the image in one clear, succint sentence. Limit verbosity."

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd', 'vlm_out', 'molmo_' + TASK)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(MODEL_DIR)


# Initialize attn_act as a list of lists, one per layer
num_layers = 28
batch_size = 5
# Module level, as in batch_inference_savdic.py: the hook closures resolve
# attn_act globally, so a per-image rebinding inside main() would leave them
# looking at an undefined name (silently swallowed by the hook's try/except).
attn_act = [[] for _ in range(num_layers)]


# ── NSD addition: reduce raw attention to 14x14 spatial maps ───────────────────
_GRID = {}

def _load_grid():
    """Rebuild Molmo's 3x3-crop token grid from image_input_idx.pt (saved by the model)."""
    if 'idx' in _GRID:
        return _GRID['idx'], _GRID['valid']
    p = os.path.join(OUTPUT_DIR, 'image_input_idx.pt')
    img_input_idx = torch.load(p, map_location='cpu')[0].numpy()
    img_idx_all = (img_input_idx[1:]
                   .reshape(3, 3, 12, 12)
                   .transpose(0, 2, 1, 3)
                   .reshape(36, 36))
    _GRID['idx'] = img_idx_all
    _GRID['valid'] = np.where(img_idx_all > 0)
    return img_idx_all, _GRID['valid']


def reduce_to_maps(final_tensor, out_size=14):
    """(n_layers, n_heads, n_steps, seq) -> (n_layers, n_heads, out_size**2) float32."""
    import cv2
    img_idx_all, valid_idx = _load_grid()
    attn = final_tensor.numpy()
    n_layers, n_heads = attn.shape[0], attn.shape[1]
    img_positions = img_idx_all[valid_idx]
    row_valid = np.any(img_idx_all > 0, axis=1)
    col_valid = np.any(img_idx_all > 0, axis=0)

    out = np.zeros((n_layers, n_heads, out_size * out_size), dtype=np.float32)
    for L in range(n_layers):
        for H in range(n_heads):
            # sum attention over generation steps, scattered into the crop grid
            vol = np.zeros(img_idx_all.shape, dtype=np.float32)
            vol[valid_idx] = attn[L, H, :, img_positions].sum(axis=-1)
            cropped = vol[row_valid][:, col_valid]
            out[L, H] = cv2.resize(cropped, (out_size, out_size),
                                   interpolation=cv2.INTER_AREA).flatten()
    return torch.from_numpy(out)


def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def attention_hook_factory(layer_idx):
    def attention_hook(module, input, output):
        try:
            if hasattr(output, 'shape'):
                if len(output.shape) >= 3 and output.shape[2] == 1:
                    #print(f"  RETURNING decoder attn output shape: {output.shape}")
                    attn_act[layer_idx].append(output.detach().cpu())
                    #print(f"Appended activation for layer={layer_idx}: {output.shape}")
                    return output
                else:
                    #print(f"  SKIPPING encoder attn output shape: {output.shape}")
                    pass
        except Exception as e:
            print(f"[Attention Hook Exception][Layer {layer_idx}] {e}", flush=True)
            raise
        return output
    return attention_hook



def register_attention_hooks(model):
    hooks = []
    print("Registering attention hooks...")
    layer_idx = 0
    for name, module in model.named_modules():
        if "isaac_attn" in name.lower():
            print(f"Registering hook on: {name} ({module.__class__.__name__}), layer {layer_idx}")
            hooks.append(module.register_forward_hook(attention_hook_factory(layer_idx)))
            layer_idx += 1
    if not hooks:
        print("No attention modules found by name. Try adjusting the filter in register_attention_hooks().")
        pass
    return hooks

def pad_and_stack_layer(layer_acts):
    # Find the maximum seq_len_k across all activations
    max_seq_len = max(t.shape[-1] for t in layer_acts)
    num_heads = layer_acts[0].shape[1]
    num_steps = len(layer_acts)
    padded = []
    for t in layer_acts:
        pad_amt = max_seq_len - t.shape[-1]
        t_squeezed = t.squeeze(0).squeeze(-2)  # Remove batch dim: [num_heads, seq_len_k]
        if pad_amt > 0:
            t_padded = torch.nn.functional.pad(t_squeezed, (0, pad_amt), value=0)
        else:
            t_padded = t_squeezed
        padded.append(t_padded)
    # Stack along new generation step dimension: [num_steps, num_heads, max_seq_len]
    stacked = torch.stack(padded, dim=0)
    # Rearrange to [num_heads, num_steps, max_seq_len]
    stacked = stacked.permute(1, 0, 2)
    return stacked  # shape: [num_heads, num_steps, max_seq_len]



def main(local_dir, prompt="Describe this image:", max_new_tokens=32):
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    print(f"Processor patch size: {processor.image_processor.image_patch_size}")
    # NSD run: the base script leaves the model on CPU. Place it on the GPU the job
    # requests, as batch_inference_gpu.py does, or 1098 images is impractical.
    _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fp32, matching batch_inference_gpu.py and the paper's existing activations
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, trust_remote_code=True).to(_dev)
    print(f'model on {_dev}')

    #model = AutoModelForCausalLM.from_pretrained(local_dir, trust_remote_code=True, torch_dtype=torch.float32)
    model.model.save_dir = OUTPUT_DIR   # needed: writes image_input_idx.pt for the grid
    model.eval()
    print(f'Prompt: {prompt}')

    # Get image paths
    captions_path = os.path.join(OUTPUT_DIR, "generated_captions.csv")
    # Load existing captions if available
    if os.path.exists(captions_path):
        existing_df = pd.read_csv(captions_path)
        already_done = set(existing_df["image_path"].tolist())
        print(f"Found {len(already_done)} already captioned images, skipping them...")
    else:
        existing_df = pd.DataFrame(columns=["image_path", "generated_text"])
        already_done = set()

    # Filter out completed images
    meta_data_path = "/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/nsd_metadata.csv"
    df = pd.read_csv(meta_data_path)
    image_paths = [p for p in df['path'].tolist() if p not in already_done]
    print(f"Remaining images to caption: {len(image_paths)}")


    # Register hooks
    hooks = register_attention_hooks(model)
    batch_outputs = []


    for batch_num, batch_paths in enumerate(
        tqdm(batchify(image_paths, batch_size), total=len(image_paths) // batch_size + 1)
    ):
        if not batch_paths:
            continue
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        image_names = [os.path.basename(p)[:-4] for p in batch_paths]

        generated_texts = []
        for i, img in enumerate(images):
            img_name = image_names[i]
            for _layer_buf in attn_act:   # in-place: hooks hold this exact object
                _layer_buf.clear()
            print(f"\n****Processing image {img_name} in batch {batch_num} HEHE HAHA****\n")

            inputs = processor.process(images=[img], text=PROMPT)
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            ## # Get masking information ##
            #print("input_ids:", inputs['input_ids'][:40])
            #input_ids = inputs['input_ids'].cpu().numpy()
            #print("input_ids:", input_ids[:40])
            #print("input_ids:", input_ids)
            #print("input_ids shape:", input_ids.shape)
            #diffs = np.diff(input_ids)
            #print("diffs between tokens:", diffs)
            #print(processor.tokenizer.decode(input_ids[0]))
            #vision_indices = [i for i, d in enumerate(diffs[0]) if abs(d) == 2]
            #print("Indices where abs(diff) == 2:", vision_indices)

            with torch.no_grad():
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
        
            # output shape: [batch_size, seq_len]
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f'Batch {batch_num}, Image {i}: {generated_text}')
            batch_outputs.append((batch_paths[i], generated_text))
            # Print a preview
            print(f"{i}: {batch_paths[i]} -> {generated_text}")

            # Extract activations for all layers
            '''
            for i, act in enumerate(attn_act):
                for j, t in enumerate(act):
                    print(f"Layer {i}, Step {j}: {t.shape}")
            '''
            all_layers_padded = []
            for layer_acts in attn_act:
                out = pad_and_stack_layer(layer_acts)
                all_layers_padded.append(out)
            
            final_tensor = torch.stack(all_layers_padded, dim=0)
            print("Final tensor shape:", final_tensor.shape)
            #print("Activations shape:", activations.shape)
            # NSD run: reduce to per-layer/per-head 14x14 spatial maps in-loop.
            # Saving the raw (28, 28, n_tok, 1028) tensor would be ~55 MB/image, i.e.
            # ~120 GB over 1098 images x 2 tasks. The reduction is the same one used in
            # tribe/scripts/2_feature_extraction.py:_extract_molmo_map.
            activation_save_path = os.path.join(OUTPUT_DIR, f'attnmap_img_{img_name}.pt')
            maps = reduce_to_maps(final_tensor.to(torch.float32))
            torch.save(maps.to(torch.float16), activation_save_path)
            print(f"Saved reduced maps {tuple(maps.shape)} -> {activation_save_path}")
            
            ## when saving with slurm 
            #torch.save(final_tensor.to(torch.float16), activation_save_path, _use_new_zipfile_serialization=False)
            print(f"Saved activations for image {i} to {activation_save_path}")

        # Save generated captions
        # Append to CSV instead of overwriting
        new_df = pd.DataFrame(batch_outputs, columns=['image_path', 'generated_text'])
        new_df.to_csv(captions_path, mode='a', header=not os.path.exists(captions_path), index=False)
        print(f"Appended {len(batch_outputs)} rows to {captions_path}")
        batch_outputs = []  # reset after writing batch





    # Clean up hooks
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    main(MODEL_DIR, PROMPT, MAX_NEW_TOKENS)