#!/bin/bash
#SBATCH --job-name=tb_score
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/tb_score_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/tb_score_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00


source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_HOME=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/data/torch_home
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines
python -u score_tierB.py
echo "DONE_OK"
