#!/usr/bin/env bash
# Scaling sweep: same data, canvas and step budget for every model size.
# Jobs form one chain on the single V100: train -> eval -> next train -> ... Evals must
# not overlap the next training run, or both compete for GPU memory (OOM).
# base/large use activation checkpointing to fit batch 64 at 2048 tokens in 32 GB.
#
#   AFTER=<jobid> bash slurm/scale_sweep.sh [key=value ...]
#     AFTER: optional job to wait for first; overrides apply to every size.
set -euo pipefail
cd "$(dirname "$0")/.."
prev=${AFTER:-}
for size in tiny small base large; do
  ckpt=false; [[ $size == base || $size == large ]] && ckpt=true
  jid=$(sbatch --parsable --job-name=artscii-sweep ${prev:+--dependency=afterany:$prev} \
    slurm/train.sbatch configs/uncond_small.yaml name=sweep_$size size=$size grad_ckpt=$ckpt "$@")
  prev=$(sbatch --parsable --job-name=artscii-sweep --dependency=afterok:$jid slurm/eval.sbatch runs/sweep_$size)
done
squeue -u "$USER"
