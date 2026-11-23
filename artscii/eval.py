"""Cheap quantitative checks for unconditional samples.

    python -m artscii.eval RUN_DIR [--n 256] [--steps 128] [--remask random] [--temperature 1.0]

  * nn_dist:   per sample, fraction of differing cells vs its nearest training grid
               (Hamming, top-left aligned). ~0 means the model copied a training piece.
  * empty:     fraction of samples with almost no ink (collapse to blank canvases).
  * char_kl:   KL(train || samples) over ink-character frequencies.
  * mean_ink:  average number of non-blank cells, samples vs train.
"""
import argparse
import json
from pathlib import Path

import torch

from artscii.data.dataset import PAD, VOCAB_SIZE, AsciiArtDataset
from artscii.diffusion.llada import sample
from artscii.sample import load_model


def crop_top_left(grids, H, W):
    """Shift each grid so its ink starts at (0, 0); comparisons ignore placement."""
    out = torch.full_like(grids, PAD)
    for i, g in enumerate(grids.view(-1, H, W)):
        ink = (g != 0) & (g != PAD)
        if ink.any():
            r, c = ink.any(1).nonzero()[0, 0], ink.any(0).nonzero()[0, 0]
            out[i].view(H, W)[: H - r, : W - c] = g[r:, c:]
    return out


def char_hist(grids):
    ink = grids[(grids != 0) & (grids != PAD)]
    h = torch.bincount(ink, minlength=VOCAB_SIZE).float() + 1  # add-one smoothing
    return h / h.sum()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--remask", default="random")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    model, cfg = load_model(args.run_dir)
    H, W = cfg["canvas_h"], cfg["canvas_w"]
    train = AsciiArtDataset(f"{cfg['data_dir']}/train.jsonl", H, W)
    train_grids = torch.stack([train[i]["tokens"] for i in range(len(train))]).cuda()
    samples = torch.cat([sample(model, 64, H * W, args.steps, args.remask, args.temperature)
                         for _ in range(args.n // 64)])
    samples = crop_top_left(samples, H, W)

    nn = torch.cat([(s[None] != train_grids).float().mean(1).min()[None] for s in samples])
    ink = lambda g: ((g != 0) & (g != PAD)).sum(1).float()
    p, q = char_hist(train_grids), char_hist(samples)
    result = dict(remask=args.remask, steps=args.steps, temperature=args.temperature, n=len(samples),
                  nn_dist_mean=nn.mean().item(), nn_dist_min=nn.min().item(),
                  frac_near_copies=(nn < 0.02).float().mean().item(),
                  empty=(ink(samples) < 15).float().mean().item(),
                  char_kl=(p * (p / q).log()).sum().item(),
                  mean_ink_samples=ink(samples).mean().item(),
                  mean_ink_train=ink(train_grids).mean().item())
    out = Path(args.run_dir) / f"eval_{args.remask}_{args.steps}_t{args.temperature}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
