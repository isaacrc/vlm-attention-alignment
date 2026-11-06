#!/bin/bash
#SBATCH --job-name=fig4
#SBATCH --output=fig4_%j.out
#SBATCH --error=fig4_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:14:00
set -e
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_roi_atlas.py --mode grouped --width 23 --panel-letter A
python plot_nsd_manuscript_figs.py --head-rule cvbehav --tasks social --regional-only --attn-only --panel-letter B --ymax 0.08 --no-stars
mv /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/regional/nsd_regional_attnonly_cvbehav_social.png \
   /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/regional/nsd_regional_attnonly_social_nostars.png
python plot_nsd_manuscript_figs.py --head-rule cvbehav --tasks social --regional-only --attn-only --panel-letter B --ymax 0.08
mv /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/regional/nsd_regional_attnonly_cvbehav_social.png \
   /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/regional/nsd_regional_attnonly_social_stars.png
echo DONE_OK
