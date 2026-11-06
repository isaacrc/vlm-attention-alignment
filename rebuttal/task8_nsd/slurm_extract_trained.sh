#!/bin/bash
#SBATCH --job-name=nsd_attn
#SBATCH --output=nsd_attn_%A_%a.out
#SBATCH --error=nsd_attn_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-7

# 4 trained captioning models x 2 tasks over the 1098 NSD stimuli.
# Timing basis: the 200-image runs took minutes for these models; 2h is ~5x headroom.

MODELS=(CNN_LSTM CNN_LSTM ViT_LSTM ViT_LSTM CNN_Transformer CNN_Transformer ViT_Transformer ViT_Transformer)
TASKS=(describe social describe social describe social describe social)

M=${MODELS[$SLURM_ARRAY_TASK_ID]}
T=${TASKS[$SLURM_ARRAY_TASK_ID]}

module purge
module load anaconda3/2022.5
conda activate cans-env
# The compute nodes' /lib64/libstdc++.so.6 predates GLIBCXX_3.4.30, which cans-env's torch
# needs; the env ships a newer one, so put it ahead of the system copy.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd

echo "=== $M / $T on $(hostname) ==="
python extract_nsd_trained.py --model "$M" --task "$T"
