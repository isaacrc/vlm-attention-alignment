#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/hf_upload_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate cans-env

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment
python upload_weights_to_hf.py
