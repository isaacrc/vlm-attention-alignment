#!/bin/bash
#SBATCH --job-name=nsd_enc_r2sup
#SBATCH --output=nsd_enc_r2score_supplement_%A_%a.out
#SBATCH --error=nsd_enc_r2score_supplement_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --array=1-8

# ROBUSTNESS SUPPLEMENT -- NOT a final analysis.
# Re-fits the identical banded-ridge pipeline but scores with himalaya's r2_score
# (true coefficient of determination) instead of signed squared Pearson r. Variance
# partitioning is defined on R2, so this checks whether the unique-attention-over-CLIP
# ordering and floor-clearance survive the metric choice. Output goes to
# encoding_r2score_supplement/ so it can never be confused with the final encoding/ set.
# Best-head rule, to match the currently featured figures (LSTMs are head-invariant).

module purge
module load anaconda3/2022.5
conda activate tribe_enc

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd

echo "=== [R2SCORE SUPPLEMENT] subject $SLURM_ARRAY_TASK_ID (best head) on $(hostname) ==="
python fit_encoding.py --subject "$SLURM_ARRAY_TASK_ID" --head-rule best --metric r2
