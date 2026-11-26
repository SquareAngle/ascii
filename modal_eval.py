"""Evaluate Modal-trained runs on Modal; the counterpart of slurm/eval.sbatch.

For each run: artscii.eval with random remasking at temperatures 0 / 0.7 / 1.0 and
low-confidence remasking at 0.7, then artscii.sample with a trajectory. Results are
written next to the run on the volume (runs/<name>/eval_*.json, sample_*.txt).

    modal run modal_eval.py --runs uncond_small_modal
    modal run modal_eval.py --runs sweep_tiny,sweep_small,sweep_base,sweep_large
    modal run modal_eval.py --runs <name> --extra "--n 512"   # extra artscii.eval args
    modal run modal_eval.py --runs cond_small --prompts "a cat;a sailboat" --guidance 2
        # conditional runs: also sample each prompt (artscii.eval itself measures the
        # unconditional / null-condition samples)
    modal volume get artscii runs/<name> ./runs/

Only runs trained through modal_train.py work: eval reads data_dir from the
checkpoint's config, which must point at the volume (/vol/processed).
"""
import os

import modal

GPU = "A10G"  # sampling only; 24 GB is plenty for every preset up to large

app = modal.App("artscii-eval")
VOLUME = os.environ.get("ARTSCII_VOLUME", "artscii")  # dataset, captions, runs
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)  # text encoder weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy", "pyyaml", "tqdm", "matplotlib", "pillow",
                 "transformers==4.57.1", "sentencepiece")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("artscii")
)

SETTINGS = [("random", "0"), ("random", "0.7"), ("random", "1.0"), ("low_confidence", "0.7")]


@app.function(image=image, gpu=GPU, volumes={"/vol": vol, "/hf": hf_vol}, timeout=4 * 3600, memory=32768)
def evaluate(name: str, extra: list[str], prompts: list[str], guidance: float):
    import subprocess

    vol.reload()  # pick up the latest checkpoint
    run_dir = f"/vol/runs/{name}"
    for remask, temp in SETTINGS:
        subprocess.run(["python", "-m", "artscii.eval", run_dir, "--remask", remask,
                        "--temperature", temp, *extra], check=True, cwd="/root")
    subprocess.run(["python", "-m", "artscii.sample", run_dir, "--temperature", "0.7",
                    "--trajectory"], check=True, cwd="/root")
    for prompt in prompts:  # text-conditioned runs: one file per prompt
        subprocess.run(["python", "-m", "artscii.sample", run_dir, "--temperature", "1.0",
                        "--prompt", prompt, "--guidance", str(guidance)], check=True, cwd="/root")
    vol.commit()


@app.local_entrypoint()
def main(runs: str, extra: str = "", prompts: str = "", guidance: float = 2.0):
    names = [r.strip() for r in runs.split(",") if r.strip()]
    ps = [p.strip() for p in prompts.split(";") if p.strip()]
    list(evaluate.starmap([(n, extra.split(), ps, guidance) for n in names]))
