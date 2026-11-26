"""Train on Modal (cloud GPUs) instead of the SLURM cluster.

One-time setup (laptop or cluster login node):
    pip install modal && modal setup
    modal volume put ascii ~/artscii_data/processed /processed

Train (same overrides as `python -m artscii.train --set ...`, comma-separated):
    modal run --detach modal_train.py --overrides "name=uncond_small_modal"
    modal run --detach modal_train.py --overrides "name=big,size=large,batch_size=128"

Scaling sweep, one GPU per size in parallel (the cluster ran them one after another):
    modal run --detach modal_train.py --sizes tiny,small,base,large

Fetch results (log.jsonl, samples, config; checkpoints live under ckpt/<name>/):
    modal volume get ascii runs/<name> ./runs/
    modal volume get ascii ckpt/<name>/ckpt.pt ./runs/<name>/

Live curves on Weights & Biases: create a Modal secret holding your W&B key once,
    modal secret create wandb WANDB_API_KEY=<key from wandb.ai/authorize>
and runs log to the W&B project named by `wandb_project` (set in the configs). Without
the secret, training runs exactly as before. ARTSCII_WANDB_SECRET picks another secret name.

Every run passes resume=true, so re-running the same command (after a failure or the
24 h function limit) continues from the last ckpt.pt, which train.py writes every
`sample_every` steps.
"""
import os

import modal

GPU = "A100-80GB"

app = modal.App("artscii-train")
VOLUME = os.environ.get("ARTSCII_VOLUME", "ascii")  # dataset, captions, runs
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)  # text encoder weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy", "pyyaml", "tqdm", "matplotlib", "pillow",
                 "transformers==4.57.1", "sentencepiece", "wandb==0.22.3")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("artscii")
    .add_local_dir("configs", "/root/configs")
)



def _wandb_secrets():
    """The W&B key secret if it exists, else nothing (W&B logging then stays off)."""
    secret = modal.Secret.from_name(os.environ.get("ARTSCII_WANDB_SECRET", "wandb"))
    try:
        secret.hydrate()
    except modal.exception.NotFoundError:
        return []
    return [secret]


# Redirect the cluster paths (/scratch/...) onto the volume, mounted at /vol.
VOLUME_PATHS = ["data_dir=/vol/processed", "ckpt_root=/vol/ckpt", "out_root=/vol/runs",
                "resume=true"]


@app.function(image=image, gpu=GPU, volumes={"/vol": vol, "/hf": hf_vol}, timeout=24 * 3600,
              retries=2, cpu=8, memory=32768, secrets=_wandb_secrets())
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
