#!/bin/bash
#SBATCH --job-name=blur_sweep
#SBATCH --account=graziano
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/blur_sweep_bilin_%j.out
source ~/.bashrc
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
MOLMO_BILINEAR=1 python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/blur_sweep.py
