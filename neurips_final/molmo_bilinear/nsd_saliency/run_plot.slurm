#!/bin/bash
#SBATCH --job-name=nsd_sal_plot
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate cans-env
set -eo pipefail
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency
python plot_nsd_saliency.py
