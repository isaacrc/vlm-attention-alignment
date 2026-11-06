#!/bin/bash
#SBATCH --job-name=bm2
#SBATCH --output=bm2_%j.out
#SBATCH --error=bm2_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_nsd_brainmap.py --model CNN_LSTM --task social --head-rule cvbehav --vmax-pct 90
echo DONE_OK
