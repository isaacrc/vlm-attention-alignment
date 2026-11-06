#!/bin/bash
#SBATCH --job-name=figs_v2
#SBATCH --output=figs_v2_%j.out
#SBATCH --error=figs_v2_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:14:00
set -e
module purge; module load anaconda3/2022.5; conda activate tribe_enc
REPO=/scratch/gpfs/GRAZIANO/vlm-attention-alignment
cd $REPO/rebuttal/task8_nsd
python plot_nsd_brainmap.py --model CNN_LSTM --task social --head-rule cvbehav --panel-letter A
python plot_nsd_manuscript_figs.py --head-rule cvbehav --tasks social --regional-only --panel-letter B
python plot_nsd_manuscript_figs.py --head-rule cvbehav --tasks social --raw-floor --panel-letter B
cd $REPO/rebuttal/plots
python plot_fig_head_selection_corrected.py
echo DONE_OK
