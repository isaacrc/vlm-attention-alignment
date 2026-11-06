#!/bin/bash
#SBATCH --job-name=nsd_sal_maps
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
set -eo pipefail
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency
python make_nsd_saliency.py
