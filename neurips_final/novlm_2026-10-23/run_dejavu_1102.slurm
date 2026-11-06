#!/bin/bash
#SBATCH --job-name=dejavu_1102
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/logs/%x_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:20:00
#SBATCH --account=graziano
source ~/.bashrc
conda activate cans-env
set -eo pipefail
unset MATPLOTLIBRC
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23
for f in fig3_neural_within_ceiling fig7_nsd_baselines_within_ceiling fig2_attention_maps_palette; do python run_dejavu_1102.py $f.py; done
echo DONE_OK
