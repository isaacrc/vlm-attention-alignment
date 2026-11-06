#!/bin/bash
# main_2026-10-23: Figure 2 without the VLMs, and edge energy / luminance table rows.
#SBATCH --job-name=novlm_1023
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --account=graziano

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23
python fig2_attention_maps.py
python score_edge_lum.py
echo "DONE_OK"
