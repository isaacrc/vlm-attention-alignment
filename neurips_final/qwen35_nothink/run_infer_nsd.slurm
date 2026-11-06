#!/bin/bash
# Qwen3.5 thinking-off re-extraction on the 1,098 NSD stimuli, one task per array element.
# Thinking-off on 200 COCO images took ~4.5 min/task including model load (~1.3 s/image);
# NSD images are larger (425 px), so allow ~2.5 s/image -> ~45 min, 1 h covers ~80%.
#SBATCH --job-name=qwen35_nothink_nsd
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=graziano
#SBATCH --time=01:00:00
#SBATCH --array=0-1

set -euo pipefail
module purge
module load anaconda3/2024.6
conda activate qwen3
export HF_HOME=/scratch/gpfs/GRAZIANO/hf_cache
export TRANSFORMERS_OFFLINE=1
TASKS=(describe social)
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink
python infer_qwen35_nothink.py --task ${TASKS[$SLURM_ARRAY_TASK_ID]} --nsd
echo "DONE_OK $(date)"
