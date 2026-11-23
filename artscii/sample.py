"""Sample from a checkpoint.

    python -m artscii.sample RUN_DIR [--n 16] [--steps 128] [--remask random] [--trajectory]

Writes RUN_DIR/sample_<remask>_<steps>_t<temperature>.txt (and a frame-by-frame trajectory file).
"""
import argparse
from pathlib import Path

import torch

from artscii.data.dataset import VOCAB_SIZE, render
from artscii.diffusion.llada import sample
from artscii.model.transformer import MaskPredictor


def load_model(run_dir, device="cuda"):
    ckpt = torch.load(Path(run_dir) / "ckpt.pt", map_location=device)
    cfg = ckpt["config"]
    model = MaskPredictor(VOCAB_SIZE, cfg["canvas_h"], cfg["canvas_w"], cfg["size"]).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--remask", default="random", choices=["random", "low_confidence"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--trajectory", action="store_true")
    args = ap.parse_args()

    model, cfg = load_model(args.run_dir)
    H, W = cfg["canvas_h"], cfg["canvas_w"]
    x, traj = sample(model, args.n, H * W, args.steps, args.remask, args.temperature,
                     return_trajectory=True)
    out = Path(args.run_dir) / f"sample_{args.remask}_{args.steps}_t{args.temperature}.txt"
    out.write_text("\n\n".join(render(g, H, W) for g in x))
    if args.trajectory:  # first sample, every step: the "watch it emerge" view
        frames = [f"--- step {i + 1}/{args.steps}\n" + render(t[0], H, W)
                  for i, t in enumerate(traj)]
        out.with_name(out.stem + "_trajectory.txt").write_text("\n".join(frames))
    print(out)


if __name__ == "__main__":
    main()
