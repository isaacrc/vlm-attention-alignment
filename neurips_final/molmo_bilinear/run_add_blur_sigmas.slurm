#!/bin/bash
# Add sigma 40 and 56 to the bilinear blur sweep.
#SBATCH --job-name=add_blur_sigmas
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=graziano

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear
python add_blur_sigmas.py
echo "DONE_OK"
