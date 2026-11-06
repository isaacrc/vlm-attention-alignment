#!/bin/bash
#SBATCH --job-name=nsd_attn_L
#SBATCH --output=nsd_attnL_%A_%a.out
#SBATCH --error=nsd_attnL_%A_%a.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --array=0-3

# ICLR revision (2026-09-23): re-extract ALL decoder layers for the two Transformer
# models on the NSD stimuli, so the encoding band can be the behaviourally
# cross-validated head (Social maxima are in layer 4, not layer 3).
# Timing basis: the layer-3 runs took 94-111 s each; 20 min is ~10x headroom.

MODELS=(CNN_Transformer CNN_Transformer ViT_Transformer ViT_Transformer)
TASKS=(describe social describe social)

M=${MODELS[$SLURM_ARRAY_TASK_ID]}
T=${TASKS[$SLURM_ARRAY_TASK_ID]}

module purge
module load anaconda3/2022.5
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd

echo "=== $M / $T all-layers on $(hostname) ==="
python extract_nsd_trained.py --model "$M" --task "$T" --all-layers
echo "DONE_OK"
