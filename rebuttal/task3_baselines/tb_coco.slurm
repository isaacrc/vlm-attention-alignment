#!/bin/bash
#SBATCH --job-name=tb_coco
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/tb_coco_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/tb_coco_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00


source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_HOME=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/data/torch_home
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines
python -u make_maps_coco.py
echo "DONE_OK"
