#!/bin/bash
# Qwen3.5 thinking-off re-extraction, both tasks back to back.
# Original thinking-on run took ~10 min/task for 200 images on one A100-80GB.
# Usage: sbatch --time=00:20:00 run_infer.slurm 000000157601 ...   (subset)
#        sbatch --time=00:40:00 run_infer.slurm                     (all 200)
#SBATCH --job-name=qwen35_nothink
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=graziano

set -euo pipefail
module purge
module load anaconda3/2024.6
conda activate qwen3
export HF_HOME=/scratch/gpfs/GRAZIANO/hf_cache
export TRANSFORMERS_OFFLINE=1

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink
IMG_ARGS=()
if [ $# -gt 0 ]; then IMG_ARGS=(--images "$@"); fi
for task in describe social; do
    python infer_qwen35_nothink.py --task $task "${IMG_ARGS[@]}"
done
echo "DONE_OK $(date)"
