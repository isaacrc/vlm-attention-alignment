#!/bin/bash
#SBATCH --job-name=molmo_sweep
#SBATCH --output=molmo_sweep_%j.out
#SBATCH --error=molmo_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00

# Full 28-layer x 28-head Molmo alignment sweep (Task 1, D4). CPU-only.
source ~/.bashrc
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task1_head_selection
python extract_molmo_alllayers.py
echo "DONE"
