#!/bin/bash
#SBATCH --job-name=nsd_sum_cvb
#SBATCH --output=nsd_sum_cvb_%j.out
#SBATCH --error=nsd_sum_cvb_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Summarise + plot the cvbehav (behaviourally cross-validated head) encoding fits.
# The per-subject ROI table must be rebuilt before summarize_final, or the final tables
# go stale against the refit npz files.

module purge
module load anaconda3/2022.5
conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
set -eo pipefail

python summarize_by_roi.py --head-rule cvbehav
python summarize_final.py  --head-rule cvbehav
python plot_wholebrain_clip_controlled.py --head-rule cvbehav
python plot_regional_rule_invariant.py --head-rule cvbehav
echo "DONE_OK"
