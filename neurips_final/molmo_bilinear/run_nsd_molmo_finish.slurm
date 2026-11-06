#!/bin/bash
#SBATCH --job-name=molmo_nsd_finish
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/logs/%x_%j.out
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:30:00
#SBATCH --account=graziano
set -eo pipefail
module purge; module load anaconda3/2022.5; conda activate tribe_enc
python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/nsd_molmo_finish.py
