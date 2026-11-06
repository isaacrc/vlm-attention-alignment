#!/bin/bash
# 8 subjects x 8 predictors; each task fits one predictor and its shuffled floor, both tasks.
#SBATCH --job-name=nsd_norm
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_norm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_norm/logs/%x_%A_%a.err
#SBATCH --array=0-63
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate tribe_enc
set -eo pipefail
M=(CNN_LSTM ViT_LSTM CNN_Transformer ViT_Transformer Molmo Qwen35 EdgeEnergy Luminance)
SUBJ=$(( SLURM_ARRAY_TASK_ID / 8 + 1 ))
MOD=${M[$(( SLURM_ARRAY_TASK_ID % 8 ))]}
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_norm
python fit_norm.py --subject $SUBJ --models $MOD Shuf$MOD --tasks social describe
echo DONE_OK
