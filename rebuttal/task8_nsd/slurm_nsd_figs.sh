#!/bin/bash
#SBATCH --job-name=nsd_figs
#SBATCH --output=nsd_figs_%j.out
#SBATCH --error=nsd_figs_%j.err
#SBATCH --account=graziano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
module purge
module load anaconda3/2022.5
conda activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_nsd_manuscript_figs.py --head-rule cvbehav
python plot_nsd_manuscript_figs.py --head-rule mean
echo "DONE_OK"
