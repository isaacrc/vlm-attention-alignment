#!/bin/bash
#SBATCH --job-name=nsd_clip
#SBATCH --output=nsd_clip_%j.out
#SBATCH --error=nsd_clip_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1
#SBATCH --time=00:40:00
module purge
module load anaconda3/2022.5
conda activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/scratch/gpfs/GRAZIANO/CANS/.hf_cache
export HF_HUB_OFFLINE=1
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python extract_nsd_clip.py
