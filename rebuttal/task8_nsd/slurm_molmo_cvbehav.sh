#!/bin/bash
#SBATCH --job-name=molmo_cvb
#SBATCH --output=molmo_cvb_%j.out
#SBATCH --error=molmo_cvb_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Reads the stored 28x28x196 Molmo NSD maps and writes the CV-behavioural band.
# Timing basis: ~8 it/s over 2196 images ~= 5 min; 15 min is ~3x headroom.

module purge
module load anaconda3/2022.5
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python build_molmo_cvbehav.py
echo "DONE_OK"
