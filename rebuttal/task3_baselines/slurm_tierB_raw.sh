#!/bin/bash
#SBATCH --job-name=tierB_raw
#SBATCH --output=tierB_raw_%j.out
#SBATCH --error=tierB_raw_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
module purge
module load anaconda3/2022.5
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines
python score_tierB_raw.py
echo "DONE_OK"
