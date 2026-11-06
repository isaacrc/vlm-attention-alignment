#!/bin/bash
#SBATCH --job-name=ymaxfig
#SBATCH --output=ymaxfig_%j.out
#SBATCH --error=ymaxfig_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:10:00
set -e
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_nsd_manuscript_figs.py --head-rule cvbehav --tasks social --regional-only --attn-only --panel-letter B --ymax 0.08 --no-stars
echo DONE_OK
