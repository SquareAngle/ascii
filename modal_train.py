"""Train on Modal (cloud GPUs) instead of the SLURM cluster.

One-time setup (laptop or cluster login node):
    pip install modal && modal setup
    modal volume put artscii ~/artscii_data/processed /processed

Train (same overrides as `python -m artscii.train --set ...`, comma-separated):
    modal run --detach modal_train.py --overrides "name=uncond_small_modal"
    modal run --detach modal_train.py --overrides "name=big,size=large,batch_size=128"

Scaling sweep, one GPU per size in parallel (the cluster ran them one after another):
    modal run --detach modal_train.py --sizes tiny,small,base,large

Fetch results (log.jsonl, samples, config; checkpoints live under ckpt/<name>/):
    modal volume get artscii runs/<name> ./runs/
    modal volume get artscii ckpt/<name>/ckpt.pt ./runs/<name>/

Every run passes resume=true, so re-running the same command (after a failure or the
24 h function limit) continues from the last ckpt.pt, which train.py writes every
`sample_every` steps.
"""
import modal

GPU = "A100-80GB"

app = modal.App("artscii-train")
vol = modal.Volume.from_name("artscii", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy", "pyyaml", "tqdm", "matplotlib", "pillow")
    .add_local_python_source("artscii")
    .add_local_dir("configs", "/root/configs")
)

# Redirect the cluster paths (/scratch/...) onto the volume, mounted at /vol.
VOLUME_PATHS = ["data_dir=/vol/processed", "ckpt_root=/vol/ckpt", "out_root=/vol/runs",
                "resume=true"]


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=24 * 3600,
              retries=2, cpu=8, memory=32768)
def train(config: str, overrides: list[str]):
    import os

    os.chdir("/root")
    from artscii.train import load_config, main

    vol.reload()  # see data/checkpoints written since this container started
    main(load_config(config, VOLUME_PATHS + overrides))
    vol.commit()


@app.local_entrypoint()
def main(config: str = "configs/uncond_small.yaml", overrides: str = "", sizes: str = ""):
    sets = [kv.strip() for kv in overrides.split(",") if kv.strip()]
    if not sizes:
        train.remote(config, sets)
        return
    # Sweep: name each run sweep_<size>; activation checkpointing is unnecessary at 80 GB.
    jobs = [(config, [f"name=sweep_{s}", f"size={s}", *sets])
            for s in (s.strip() for s in sizes.split(",")) if s]
    list(train.starmap(jobs))
