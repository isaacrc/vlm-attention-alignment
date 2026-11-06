#!/bin/bash
#SBATCH --job-name=fonts_1102
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/logs/%x_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:20:00
#SBATCH --account=graziano
set -eo pipefail
export MATPLOTLIBRC=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23/mplrc
module purge; module load anaconda3/2022.5
source activate tribe_enc
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
python plot_roi_atlas.py --mode grouped --width 23 --panel-letter A
conda deactivate; source activate cans-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/novlm_2026-10-23
python fig3_neural_within_ceiling.py
for f in fig1_alignment fig4_metric_suite fig5_smoothness fig_baselines_rsa; do python run_fonts_1102.py $f.py; done
echo DONE_OK
