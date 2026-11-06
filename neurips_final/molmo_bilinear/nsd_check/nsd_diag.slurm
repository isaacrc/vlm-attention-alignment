#!/bin/bash
#SBATCH --job-name=nsd_diag
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_check/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_check/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_check
python nsd_diag.py
