#!/bin/bash
# Molmo + ShufMolmo for the task(s) whose CV head changed (nsd_molmo_prepare.py writes
# them to REFIT_TASKS.txt). ~400 s per model x task (qwen35_nothink/run_nsd_fit.slurm),
# so 2-4 fits take 14-27 min per subject.
#SBATCH --job-name=molmo_nsd_fit
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/molmo_bilinear/logs/%x_%A_%a.out
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:35:00
#SBATCH --array=1-8
#SBATCH --account=graziano
set -eo pipefail
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
TASKS=$(cat _superseded_molmo_area/REFIT_TASKS.txt)
python fit_encoding.py --subject "$SLURM_ARRAY_TASK_ID" --head-rule cvbehav --models Molmo ShufMolmo --tasks $TASKS
echo DONE_OK
