#!/bin/bash
#SBATCH --job-name=headL4_rerun
#SBATCH --output=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/arch_training_deck/headL4_rerun/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/arch_training_deck/headL4_rerun/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=graziano
module purge; module load anaconda3/2024.6
conda activate cans-env
for c in CNN_Transformer ViT_Transformer; do
  python /scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/arch_training_deck/headL4_rerun/head_attention_analysis.py --condition $c --layer 4
done
