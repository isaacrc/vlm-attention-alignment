#!/bin/bash
#SBATCH --job-name=t1cv
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection/t1cv_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection/t1cv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection
python build_cvhonest_table.py
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task5_statistics
python run_stats.py --table /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/data/alignment_long_cv_honest.parquet --tag cv_honest
echo "DONE_OK"
