#!/bin/bash
#SBATCH --job-name=nsd_chain
#SBATCH --output=nsd_chain_%j.out
#SBATCH --error=nsd_chain_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --time=01:00:00

# Runs once Molmo extraction AND the first encoding pass have finished:
#   1. build Molmo attention features
#   2. resubmit the encoding array, which skips completed conditions and so picks up
#      only Molmo and Qwen
#   3. submit the exploratory overlap analysis, dependent on that encoding pass

set -e
module purge
module load anaconda3/2022.5
conda activate cans-env
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd

echo "=== building Molmo features ==="
python build_vlm_features.py --models Molmo

echo "=== resubmitting encoding for the VLM conditions ==="
ENC=$(sbatch --parsable slurm_fit_encoding.sh)
echo "encoding array: $ENC"

echo "=== queuing overlap analysis after encoding ==="
OVL=$(sbatch --parsable --dependency=afterany:$ENC slurm_fit_overlap.sh)
echo "overlap array: $OVL"
