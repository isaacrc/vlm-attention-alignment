#!/bin/bash
#SBATCH --job-name=nsd_molmo
#SBATCH --output=nsd_molmo_%A_%a.out
#SBATCH --error=nsd_molmo_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --array=0-1

# Molmo 7B-D over the 1098 NSD stimuli, both tasks.
# Measured 2.7-5.4 s/image on the 200-image runs -> 1-2 h of generation, but the
# in-loop 28x28 map reduction adds per-image CPU work, so 10 h is the safe envelope.

TASKS=(describe social)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}

module purge
module load anaconda3/2024.2
conda activate molmo

cd /scratch/gpfs/GRAZIANO/CANS/molmo/molmo/launch_scripts
echo "=== molmo / $T on $(hostname) ==="
python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/batch_inference_nsd.py --task "$T"
