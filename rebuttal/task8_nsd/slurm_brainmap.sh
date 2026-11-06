#!/bin/bash
#SBATCH --job-name=brainmap
#SBATCH --output=brainmap_%j.out
#SBATCH --error=brainmap_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
module purge; module load anaconda3/2022.5; conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
for T in social describe; do
  python plot_nsd_brainmap.py --model CNN_LSTM --task $T --head-rule cvbehav
done
echo DONE_OK
