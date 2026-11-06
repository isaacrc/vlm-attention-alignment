#!/bin/bash
#SBATCH --job-name=lmm_effcoded
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/lmm_effect_coded.py
