#!/bin/bash
#SBATCH --job-name=fig3_panels
#SBATCH --output=fig3_panels_%j.out
#SBATCH --error=fig3_panels_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -e
module purge; module load anaconda3/2022.5; conda activate tribe_enc
REPO=/scratch/gpfs/GRAZIANO/vlm-attention-alignment

# Fig 3 panel A: whole-brain variance partition, bare bold "A" drawn in-image.
cd $REPO/rebuttal/task8_nsd
python plot_nsd_brainmap.py --model CNN_LSTM --task social --head-rule cvbehav --panel-letter A

# Fig 3 panel B: regional streams, Social only, matching "B".
python plot_nsd_manuscript_figs.py --head-rule cvbehav --tasks social --regional-only --panel-letter B

# Fig 1: same script, minus the suptitle Isaac asked to drop.
cd $REPO/rebuttal/plots
python plot_fig_head_selection_corrected.py
echo DONE_OK
