#!/usr/bin/env bash
# Copy the processed dataset from the shared home dir to bwrcr740-4's local /scratch
# (no ssh to compute nodes, so the copy runs as a tiny SLURM job).
set -euo pipefail
srun -p normal -w bwrcr740-4 -n 1 -c 1 -t 10 \
  rsync -a --delete "$HOME/artscii_data/processed/" /scratch/sahir.tandon/artscii/processed/
