#!/bin/bash
#SBATCH --job-name=mapsim
#SBATCH --output=mapsim_%j.out
#SBATCH --error=mapsim_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
module purge
module load anaconda3/2022.5
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines
python map_similarity.py
echo "DONE_OK"
