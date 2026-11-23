#!/usr/bin/env bash
# Usage: sbatch slurm/train.sbatch configs/uncond_small.yaml [key=value ...]
#SBATCH --job-name=artscii
#SBATCH --partition=normal
#SBATCH --nodelist=bwrcr740-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG=$1; shift
.venv/bin/python -m artscii.train --config "$CONFIG" --set "$@"
