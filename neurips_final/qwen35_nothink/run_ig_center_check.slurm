#!/bin/bash
#SBATCH --account=graziano --mem=48G --time=00:30:00 --cpus-per-task=2 -o /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/ig_center_check.out
source ~/.bashrc; conda activate cans-env; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/ig_center_check.py
