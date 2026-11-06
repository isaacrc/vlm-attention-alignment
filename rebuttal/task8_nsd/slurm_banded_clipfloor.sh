#!/bin/bash
#SBATCH --job-name=bandclipfl
#SBATCH --output=bandclipfl_%A_%a.out
#SBATCH --error=bandclipfl_%A_%a.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=1-8
# One banded fit per subject with CLIP permuted. ~8 min typical; 0:45 is slack because
# one node in the earlier floor run was twice as slow as the rest.
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python fit_banded_clipfloor.py --subject $SLURM_ARRAY_TASK_ID --model CNN_Transformer --task social
echo DONE_OK
