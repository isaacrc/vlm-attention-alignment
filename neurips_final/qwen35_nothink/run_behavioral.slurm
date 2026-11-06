#!/bin/bash
# Qwen sweep (2 tasks x 128 heads) + CV selection + between-task + stats.
# Comparable rebuttal steps (t1build + t1cv) each finished well inside 1 h.
#SBATCH --job-name=qwen35_nothink_behav
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --account=graziano

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink
python rerun_behavioral.py
echo "DONE_OK"
