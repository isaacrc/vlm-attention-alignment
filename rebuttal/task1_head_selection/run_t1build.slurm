#!/bin/bash
#SBATCH --job-name=t1build
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection/t1build_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection/t1build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection
python build_head_table.py
echo "DONE_OK"
