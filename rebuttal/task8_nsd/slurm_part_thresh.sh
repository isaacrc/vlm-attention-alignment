#!/bin/bash
#SBATCH --job-name=partthresh
#SBATCH --output=partthresh_%j.out
#SBATCH --error=partthresh_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:18:00
set -e
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_brainmap_partition_thresh.py --model CNN_Transformer --task social --panel-letter A
echo DONE_OK
