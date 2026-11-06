#!/bin/bash
#SBATCH --job-name=tb_sal
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/tb_sal_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/tb_sal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00


source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_HOME=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/data/torch_home
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines
python -u make_maps_saliency.py
echo "DONE_OK"
