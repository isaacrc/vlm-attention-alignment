#!/bin/bash
#SBATCH --job-name=nsd_floor_s1
#SBATCH --output=nsd_floor_s1_%j.out
#SBATCH --error=nsd_floor_s1_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=02:00:00
# Subject 1 drew a slow node and timed out with 6/16 floors done. fit_encoding.py skips
# outputs that already exist, so this resumes the remaining fits.
# ~10 fits x ~450 s => ~75 min; 2:00 is ~65% utilisation, deliberately slack because the
# first attempt ran about twice as slow as the other seven subjects.
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python fit_encoding.py --subject 1 --head-rule cvbehav --models \
    ShufCNN_LSTM ShufViT_LSTM ShufCNN_Transformer ShufViT_Transformer \
    ShufMolmo ShufQwen35 ShufEdgeEnergy ShufLuminance
echo DONE_OK
