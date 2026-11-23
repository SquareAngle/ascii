"""Evaluate Modal-trained runs on Modal; the counterpart of slurm/eval.sbatch.

For each run: artscii.eval with random remasking at temperatures 0 / 0.7 / 1.0 and
low-confidence remasking at 0.7, then artscii.sample with a trajectory. Results are
written next to the run on the volume (runs/<name>/eval_*.json, sample_*.txt).

    modal run modal_eval.py --runs uncond_small_modal
    modal run modal_eval.py --runs sweep_tiny,sweep_small,sweep_base,sweep_large
    modal run modal_eval.py --runs <name> --extra "--n 512"   # extra artscii.eval args
    modal volume get artscii runs/<name> ./runs/

Only runs trained through modal_train.py work: eval reads data_dir from the
checkpoint's config, which must point at the volume (/vol/processed).
"""
import modal

GPU = "A10G"  # sampling only; 24 GB is plenty for every preset up to large

app = modal.App("artscii-eval")
vol = modal.Volume.from_name("artscii", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy", "pyyaml", "tqdm", "matplotlib", "pillow")
    .add_local_python_source("artscii")
)

SETTINGS = [("random", "0"), ("random", "0.7"), ("random", "1.0"), ("low_confidence", "0.7")]


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600, memory=32768)
def evaluate(name: str, extra: list[str]):
    import subprocess

    vol.reload()  # pick up the latest checkpoint
    run_dir = f"/vol/runs/{name}"
    for remask, temp in SETTINGS:
        subprocess.run(["python", "-m", "artscii.eval", run_dir, "--remask", remask,
                        "--temperature", temp, *extra], check=True, cwd="/root")
    subprocess.run(["python", "-m", "artscii.sample", run_dir, "--temperature", "0.7",
                    "--trajectory"], check=True, cwd="/root")
    vol.commit()


@app.local_entrypoint()
def main(runs: str, extra: str = ""):
    names = [r.strip() for r in runs.split(",") if r.strip()]
    list(evaluate.starmap([(n, extra.split()) for n in names]))
