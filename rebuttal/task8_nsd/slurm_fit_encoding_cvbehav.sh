#!/bin/bash
#SBATCH --job-name=nsd_enc_cvb
#SBATCH --output=nsd_enc_cvb_%A_%a.out
#SBATCH --error=nsd_enc_cvb_%A_%a.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --array=1-8

# ICLR revision (2026-09-23): refit with the attention band set to the head that 5-fold
# cross-validation on the FIXATION data selected, per model x task -- a single head, not a
# head average. Social-task Transformer heads are in layer 4 (L4H4, L4H8), which the old
# layer-3-only extraction could not supply; Molmo's Describe head is L8H7, not the
# in-sample argmax L10H27.
#
# Only the conditions whose band actually changes are fit here. The other 15 model x task
# cells were verified bit-identical to --head-rule best and their fits were copied, since
# the fit is deterministic given the same inputs.
#
# Timing basis: ~400 s per model x task plus ~300 s for the one-off CLIP-only fit
# => ~45 min/subject; 1:15 is ~60-80% utilisation.

module purge
module load anaconda3/2022.5
conda activate tribe_enc

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
S=$SLURM_ARRAY_TASK_ID

echo "=== subject $S (cvbehav) on $(hostname) ==="
python fit_encoding.py --subject "$S" --head-rule cvbehav \
    --models CNN_Transformer ViT_Transformer
python fit_encoding.py --subject "$S" --head-rule cvbehav \
    --models Molmo --tasks describe
echo "DONE_OK"
