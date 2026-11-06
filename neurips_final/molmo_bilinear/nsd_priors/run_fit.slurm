#!/bin/bash
# person and face priors + floors on the NSD images with a visible head, 8 subjects
#SBATCH --job-name=nsd_prior_fit
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_priors/logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_priors/logs/%x_%A_%a.err
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate tribe_enc
set -eo pipefail
M=(PriorPerson PriorFace ShufPriorPerson ShufPriorFace)
SUBJ=$(( SLURM_ARRAY_TASK_ID / 4 + 1 ))
MOD=${M[$(( SLURM_ARRAY_TASK_ID % 4 ))]}
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_priors
export NSD_ENC_DIR=encoding_norm_priors_face
export NSD_SUBSET=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/attn/face_subset_nsdIds.npy
python fit_subset.py --subject $SUBJ --models $MOD --tasks social describe
echo DONE_OK
