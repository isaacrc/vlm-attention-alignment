#!/bin/bash
#SBATCH --job-name=clip_floor
#SBATCH --output=clip_floor_%A_%a.out
#SBATCH --error=clip_floor_%A_%a.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=1-8
# One CLIP-only fit per subject on permuted embeddings; ~5-8 min each, 0:40 is slack
# because the first per-model floor run showed one node can be twice as slow.
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python fit_clip_floor.py --subject $SLURM_ARRAY_TASK_ID
echo DONE_OK
