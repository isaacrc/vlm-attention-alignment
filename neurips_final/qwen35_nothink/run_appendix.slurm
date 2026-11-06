#!/bin/bash
# Winner's curse + center-controlled + metric suite (each ~2-3 min in the rebuttal),
# then Fig 5 + entropy (archived job asked for 15 min). 45 min covers all four.
#SBATCH --job-name=qwen35_nothink_appx
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --account=graziano

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink
python rerun_appendix.py
python final_fig5_entropy.py
echo "DONE_OK"
