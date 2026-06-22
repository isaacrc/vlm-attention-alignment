"""
upload_weights_to_hf.py
-----------------------
Upload all training-trajectory checkpoints for all 4 models to HuggingFace.
Run on a compute node (CPU is fine, this is I/O bound):
    sbatch slurm_upload_hf.sh
Or directly:
    conda run -n cans-env python upload_weights_to_hf.py

HuggingFace repo: isaacrc/vlm-attention-alignment
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file

api = HfApi()
REPO_ID = "isaacrc/vlm-attention-alignment"
REPO_TYPE = "model"

CANS = Path("/scratch/gpfs/GRAZIANO/CANS")
FRM = CANS / "group_analysis/learning_to_attend/social_replication/final_replicated_model"
ARCH = CANS / "architecture_manipulation"

# Map: (local_dir, hf_path_prefix)
UPLOADS = [
    # CNN-LSTM — all 4 phases
    (FRM / "phase1_improvements",        "cnn_lstm/phase1"),
    (FRM / "phase2_improvements_bs32",   "cnn_lstm/phase2"),
    (FRM / "social_improvements_bs32",   "cnn_lstm/social_unfrozen"),
    (FRM / "social_improvements_bs32_frozen", "cnn_lstm/social_frozen"),

    # CNN-Transformer
    (ARCH / "CNN_Transformer/checkpoints/phase1",         "cnn_transformer/phase1"),
    (ARCH / "CNN_Transformer/checkpoints/phase2",         "cnn_transformer/phase2"),
    (ARCH / "CNN_Transformer/checkpoints/social_unfrozen","cnn_transformer/social_unfrozen"),
    (ARCH / "CNN_Transformer/checkpoints/social_frozen",  "cnn_transformer/social_frozen"),

    # ViT-LSTM
    (ARCH / "ViT_LSTM/checkpoints/phase1",         "vit_lstm/phase1"),
    (ARCH / "ViT_LSTM/checkpoints/phase2",         "vit_lstm/phase2"),
    (ARCH / "ViT_LSTM/checkpoints/social_unfrozen","vit_lstm/social_unfrozen"),
    (ARCH / "ViT_LSTM/checkpoints/social_frozen",  "vit_lstm/social_frozen"),

    # ViT-Transformer
    (ARCH / "ViT_Transformer/checkpoints/phase1",         "vit_transformer/phase1"),
    (ARCH / "ViT_Transformer/checkpoints/phase2",         "vit_transformer/phase2"),
    (ARCH / "ViT_Transformer/checkpoints/social_unfrozen","vit_transformer/social_unfrozen"),
    (ARCH / "ViT_Transformer/checkpoints/social_frozen",  "vit_transformer/social_frozen"),
]

total_uploaded = 0
total_skipped = 0

for local_dir, hf_prefix in UPLOADS:
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"[SKIP] {local_dir} does not exist")
        continue

    ckpts = sorted(local_dir.glob("*.pth.tar"))
    print(f"\n--- {hf_prefix} ({len(ckpts)} files) ---")

    for ckpt in ckpts:
        hf_path = f"{hf_prefix}/{ckpt.name}"
        print(f"  Uploading {ckpt.name} → {hf_path} ...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=hf_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print("done")
            total_uploaded += 1
        except Exception as e:
            print(f"ERROR: {e}")

print(f"\nDone. Uploaded {total_uploaded} files, skipped {total_skipped}.")
print(f"View at: https://huggingface.co/{REPO_ID}")
