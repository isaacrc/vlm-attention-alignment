#!/bin/bash
#SBATCH --job-name=nsd_ovl
#SBATCH --output=nsd_ovl_%A_%a.out
#SBATCH --error=nsd_ovl_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-8

# Exploratory: apply each subject's shared-1000 model, frozen, to their overlap images.
# One fit per model x task (no CV, unlike the primary analysis) at ~150 s each; 12
# conditions per subject is ~30 min, so 4 h is ample.

module purge
module load anaconda3/2022.5
conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python fit_overlap.py --subjects "$SLURM_ARRAY_TASK_ID" \
    --models CNN_LSTM ViT_LSTM CNN_Transformer ViT_Transformer Molmo Qwen35 Luminance Shuffled
