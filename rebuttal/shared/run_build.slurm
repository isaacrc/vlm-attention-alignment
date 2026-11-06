#!/bin/bash
#SBATCH --job-name=reb_build
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00

# Shared alignment table build (caches ranked humans, ceilings, base best-head table).
source ~/.bashrc
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/shared
python build_alignment_table.py
echo "DONE"
