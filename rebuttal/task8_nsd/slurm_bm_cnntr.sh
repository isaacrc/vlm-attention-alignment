#!/bin/bash
#SBATCH --job-name=bm_cnntr
#SBATCH --output=bm_cnntr_%j.out
#SBATCH --error=bm_cnntr_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:15:00
set -e
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_nsd_brainmap.py --model CNN_Transformer --task social --head-rule cvbehav --panel-letter A
echo DONE_OK
