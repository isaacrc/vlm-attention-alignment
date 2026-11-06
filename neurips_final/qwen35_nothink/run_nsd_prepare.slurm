#!/bin/bash
# Backup + build Qwen35 NSD features + floor: 2 x 1,098 small .pt reads, a few minutes.
#SBATCH --job-name=qwen35_nsd_prep
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:15:00
#SBATCH --account=graziano
set -eo pipefail
module purge; module load anaconda3/2022.5; conda activate tribe_enc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/nsd_prepare.py
echo DONE_OK
