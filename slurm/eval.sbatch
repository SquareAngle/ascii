#!/usr/bin/env bash
# Usage: sbatch slurm/eval.sbatch runs/<name> [extra artscii.eval args]
# Evaluates random remasking at 3 temperatures plus low-confidence, and writes a trajectory.
#SBATCH --job-name=artscii-eval
#SBATCH --partition=normal
#SBATCH --nodelist=bwrcr740-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN=$1; shift
for temp in 0 0.7 1.0; do
  .venv/bin/python -m artscii.eval "$RUN" --remask random --temperature $temp "$@"
done
.venv/bin/python -m artscii.eval "$RUN" --remask low_confidence --temperature 0.7 "$@"
.venv/bin/python -m artscii.sample "$RUN" --temperature 0.7 --trajectory
