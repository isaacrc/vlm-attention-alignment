#!/bin/bash
# Qwen35 + ShufQwen35, both tasks, per subject. slurm_fit_permodel_floor.sh measured
# ~400 s per model x task -> 4 fits ~27 min/subject; 35 min is ~80% utilisation.
#SBATCH --job-name=qwen35_nsd_fit
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/qwen35_nothink/logs/%x_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:35:00
#SBATCH --array=1-8
#SBATCH --account=graziano
set -eo pipefail
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python fit_encoding.py --subject "$SLURM_ARRAY_TASK_ID" --head-rule cvbehav --models Qwen35 ShufQwen35
echo DONE_OK
