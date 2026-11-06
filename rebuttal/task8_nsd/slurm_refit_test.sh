#!/bin/bash
#SBATCH --job-name=refit_test
#SBATCH --output=refit_test_%j.out
#SBATCH --error=refit_test_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=00:20:00
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python refit_test.py
