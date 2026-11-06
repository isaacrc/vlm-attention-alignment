#!/bin/bash
# Wait for the best-head fits (SLURM 11578654) to finish, then run the summarize + plot
# pipeline for --head-rule best. Prints clear terminal markers on success or failure.
set -uo pipefail
JOB=11578654
DIR=/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd
cd "$DIR"

# 1. poll until no array task of $JOB remains in the queue
while squeue -j "$JOB" -h -o "%i" 2>/dev/null | grep -q .; do
    sleep 60
done
echo "BESTHEAD: job $JOB left the queue"

# 2. verify the fits landed (expect 10 models x 2 tasks x 8 subjects = 160)
n=$(ls encoding/r2_*_best.npz 2>/dev/null | wc -l)
echo "BESTHEAD: found $n / 160 best-head npz files"
if [ "$n" -lt 160 ]; then
    echo "BESTHEAD: FAILED — incomplete fits, not plotting. Check nsd_enc_best_*.err"
    exit 1
fi

# 3. env + pipeline
module purge 2>/dev/null
module load anaconda3/2022.5 2>/dev/null
source activate tribe_enc 2>/dev/null || conda activate tribe_enc 2>/dev/null
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "BESTHEAD: summarize_by_roi..."
python summarize_by_roi.py --head-rule best || { echo "BESTHEAD: FAILED at summarize_by_roi"; exit 2; }
echo "BESTHEAD: summarize_final..."
python summarize_final.py  --head-rule best || { echo "BESTHEAD: FAILED at summarize_final"; exit 3; }
echo "BESTHEAD: plot_story_panels..."
python plot_story_panels.py --head-rule best || { echo "BESTHEAD: FAILED at plot"; exit 4; }
echo "BESTHEAD: plot_manuscript_identical..."
python plot_manuscript_identical.py --head-rule best || { echo "BESTHEAD: FAILED at manuscript plot"; exit 5; }

echo "BESTHEAD: DONE — figures written:"
ls -1 /scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/*_best.png
