#!/bin/bash
#SBATCH --job-name=nsd_enc
#SBATCH --output=nsd_enc_%A_%a.out
#SBATCH --error=nsd_enc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-8

# One job per NSD subject; each fits every available model x task.
# Measured ~110 s per banded CV fold on 1000 images x 5054 vertices -> ~12 min per
# model x task, x12 conditions = ~2.5 h. 6 h keeps utilisation near 80% if all six
# models (including the two VLMs) are present.

module purge
module load anaconda3/2022.5
conda activate tribe_enc

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd

echo "=== subject $SLURM_ARRAY_TASK_ID on $(hostname) ==="
python fit_encoding.py --subject "$SLURM_ARRAY_TASK_ID" --head-rule mean
