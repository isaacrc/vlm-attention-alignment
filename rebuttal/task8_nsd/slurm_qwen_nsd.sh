#!/bin/bash
#SBATCH --job-name=nsd_qwen
#SBATCH --output=nsd_qwen_%A_%a.out
#SBATCH --error=nsd_qwen_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --account=graziano
#SBATCH --time=03:00:00
#SBATCH --array=0-1

# Qwen3.5-9B over the 1098 NSD stimuli, both tasks.
# Measured 2.8 s/image on the 200-image runs -> ~0.9 h. 3 h covers model load plus the
# larger 425x425 inputs at ~80% utilisation. Env/account match scripts/infer_qwen35_*.slurm.

TASKS=(describe social)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}

module purge
module load anaconda3/2024.6
conda activate qwen3

cd /scratch/gpfs/GRAZIANO/CANS/vlm-probing
export HF_HOME=/scratch/gpfs/GRAZIANO/hf_cache

echo "=== qwen35 / $T on $(hostname) ==="
python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/infer_qwen35_nsd.py --task "$T"
