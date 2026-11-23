"""Plot the scaling sweep: best val loss and memorization vs. parameter count.

    python -m artscii.plot_scaling [runs/sweep_*]   ->  runs/scaling.png + table on stdout
"""
import json
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from artscii.data.dataset import VOCAB_SIZE
from artscii.model.transformer import MaskPredictor


def summarize(run: Path) -> dict:
    cfg = yaml.safe_load(open(run / "config.yaml"))
    log = [json.loads(l) for l in open(run / "log.jsonl")]
    vals = [r["val_loss"] for r in log if "val_loss" in r]
    params = MaskPredictor(VOCAB_SIZE, cfg["canvas_h"], cfg["canvas_w"], cfg["size"]).num_params()
    ev = run / "eval_random_128_t1.0.json"
    ev = json.loads(ev.read_text()) if ev.exists() else {}
    return dict(run=run.name, size=cfg["size"], params_m=params / 1e6,
                best_val=min(vals) if vals else None, final_val=vals[-1] if vals else None,
                nn_dist=ev.get("nn_dist_mean"), near_copies=ev.get("frac_near_copies"))


def main():
    runs = [Path(p) for p in (sys.argv[1:] or sorted(glob("runs/sweep_*")))]
    rows = sorted((summarize(r) for r in runs), key=lambda r: r["params_m"])
    for r in rows:
        print(r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = [r["params_m"] for r in rows]
    ax[0].plot(x, [r["best_val"] for r in rows], "o-", label="best val")
    ax[0].plot(x, [r["final_val"] for r in rows], "s--", label="final val")
    ax[0].set(xscale="log", xlabel="params (M)", ylabel="val loss (ELBO bound)")
    ax[0].legend()
    ax[1].plot(x, [r["nn_dist"] for r in rows], "o-")
    ax[1].set(xscale="log", xlabel="params (M)", ylabel="mean NN Hamming dist to train",
              title="lower = closer to copying")
    for a, r in zip([ax[0]] * len(rows), rows):
        a.annotate(r["size"], (r["params_m"], r["best_val"] or 0))
    fig.tight_layout()
    fig.savefig("runs/scaling.png", dpi=120)
    print("wrote runs/scaling.png")


if __name__ == "__main__":
    main()
