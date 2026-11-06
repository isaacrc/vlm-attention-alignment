#!/bin/bash
#SBATCH --job-name=t1molmo
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection/t1molmo_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection/t1molmo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection
python extract_molmo_alllayers.py
echo "DONE_OK"
