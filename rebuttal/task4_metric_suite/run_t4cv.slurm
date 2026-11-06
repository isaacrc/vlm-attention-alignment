#!/bin/bash
#SBATCH --job-name=t4cv
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task4_metric_suite/t4cv_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task4_metric_suite/t4cv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

source ~/.bashrc
conda activate cans-env
set -eo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task4_metric_suite
python run_metric_suite_cv.py
echo "DONE_OK"
