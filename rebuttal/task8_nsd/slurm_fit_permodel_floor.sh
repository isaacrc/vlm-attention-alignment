#!/bin/bash
#SBATCH --job-name=nsd_floor
#SBATCH --output=nsd_floor_%A_%a.out
#SBATCH --error=nsd_floor_%A_%a.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:20:00
#SBATCH --array=1-8

# NeurIPS camera-ready (2026-10-06): one permutation floor PER MODEL, replacing the single
# CNN-LSTM-derived Shuffled floor that had been subtracted from every model. See
# make_shuffled_permodel.py for why. 8 floors x 2 tasks per subject.
#
# Timing basis: ~400 s per model x task => ~1 h 50 m/subject; 2:20 is ~80% utilisation.

module purge
module load anaconda3/2022.5
conda activate tribe_enc

cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
S=$SLURM_ARRAY_TASK_ID
echo "=== subject $S per-model floors on $(hostname) ==="
python fit_encoding.py --subject "$S" --head-rule cvbehav --models \
    ShufCNN_LSTM ShufViT_LSTM ShufCNN_Transformer ShufViT_Transformer \
    ShufMolmo ShufQwen35 ShufEdgeEnergy ShufLuminance
echo "DONE_OK"
