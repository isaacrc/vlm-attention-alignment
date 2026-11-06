#!/bin/bash
#SBATCH --job-name=nsd_enc_best
#SBATCH --output=nsd_enc_best_%A_%a.out
#SBATCH --error=nsd_enc_best_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-8

# Best-head variant: attention band uses the paper's selected head (the one used in the
# behavioral fixation-alignment analysis), not the mean over the analysed layer.
# LSTMs have a single soft-attention map, so best == mean for them; only the four
# Transformer/VLM models change.

module purge
module load anaconda3/2022.5
conda activate tribe_enc

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd

echo "=== subject $SLURM_ARRAY_TASK_ID (best head) on $(hostname) ==="
python fit_encoding.py --subject "$SLURM_ARRAY_TASK_ID" --head-rule best
