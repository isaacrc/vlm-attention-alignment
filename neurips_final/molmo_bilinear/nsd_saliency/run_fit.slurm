#!/bin/bash
# normalized encoding fits (nsd_norm/fit_norm.py) for the three saliency maps, 8 subjects
#SBATCH --job-name=nsd_sal_fit
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency/logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_saliency/logs/%x_%A_%a.err
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate tribe_enc
set -eo pipefail
M=(SalItti SalGBVS SalSR)
SUBJ=$(( SLURM_ARRAY_TASK_ID / 3 + 1 ))
MOD=${M[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_norm
export NSD_ENC_DIR=encoding_norm_saliency
python fit_norm.py --subject $SUBJ --models $MOD --tasks social describe
echo DONE_OK
