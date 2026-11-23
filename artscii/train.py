"""Train the unconditional masked-diffusion model.

    python -m artscii.train --config configs/uncond_small.yaml [--set key=value ...]

Writes to runs/<name>/ (shared home, readable from the login node): config.yaml,
log.jsonl, samples_<step>.txt, and ckpt.pt -> a symlink to the checkpoint on /scratch.
"""
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from artscii.data.dataset import VOCAB_SIZE, AsciiArtDataset, collate, render
from artscii.diffusion.llada import diffusion_loss, sample
from artscii.model.transformer import MaskPredictor


@dataclass
class Config:
    name: str = "uncond_small"
    data_dir: str = "/scratch/sahir.tandon/artscii/processed"
    out_root: str = "runs"
    ckpt_root: str = "/scratch/sahir.tandon/artscii/runs"
    canvas_h: int = 32
    canvas_w: int = 64
    size: str = "small"
    dropout: float = 0.1
    grad_ckpt: bool = False
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: int = 1000
    max_steps: int = 50_000
    ema_decay: float = 0.999
    flip_p: float = 0.5
    shift_p: float = 0.5
    blank_weight: float = 1.0
    overfit_n: int = 0  # >0: train on the first n pieces only, no augmentation
    eval_every: int = 1000
    sample_every: int = 5000
    sample_steps: int = 128
    remask: str = "random"
    sample_temperature: float = 0.7  # temp 1 mixes pieces; temp 0 is deterministic per order
    num_workers: int = 6
    seed: int = 0
    resume: bool = False  # continue from this run's ckpt.pt (model, EMA, optimizer, step)


def parse_value(typ, v: str):
    if typ is bool:
        return v.lower() in ("1", "true", "yes")
    return typ(v)


def load_config(path, overrides) -> Config:
    cfg = yaml.safe_load(open(path)) if path else {}
    types = {f.name: f.type for f in fields(Config)}
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(types[k], v)
    return Config(**cfg)


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * step / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    """Val loss with a fixed masking seed, so numbers are comparable across steps/runs."""
    model.eval()
    gen = torch.Generator(device).manual_seed(0)
    total, n = 0.0, 0
    for batch in loader:
        x = batch["tokens"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            total += diffusion_loss(model, x, generator=gen).item() * len(x)
        n += len(x)
    return total / max(1, n)


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    run = Path(cfg.out_root) / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    ckpt_path = Path(cfg.ckpt_root) / cfg.name / "ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not (run / "ckpt.pt").is_symlink():
        (run / "ckpt.pt").symlink_to(ckpt_path)

    H, W = cfg.canvas_h, cfg.canvas_w
    overfit = cfg.overfit_n > 0
    train_set = AsciiArtDataset(f"{cfg.data_dir}/train.jsonl", H, W,
                                0 if overfit else cfg.flip_p, 0 if overfit else cfg.shift_p,
                                limit=cfg.overfit_n or None)
    val_set = train_set if overfit else AsciiArtDataset(f"{cfg.data_dir}/val.jsonl", H, W)
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, drop_last=not overfit,
                              collate_fn=collate, num_workers=cfg.num_workers,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, 256, collate_fn=collate, num_workers=2)

    model = MaskPredictor(VOCAB_SIZE, H, W, cfg.size, cfg.dropout, cfg.grad_ckpt).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98),
                            weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler()
    step = 0
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.module.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"resumed from step {step}", flush=True)
    print(f"{cfg.name}: {model.num_params() / 1e6:.1f}M params, "
          f"{len(train_set)} train / {len(val_set)} val pieces, canvas {W}x{H}", flush=True)

    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    while step < cfg.max_steps:
        for batch in train_loader:
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            x = batch["tokens"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = diffusion_loss(model, x, blank_weight=cfg.blank_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update_parameters(model)
            step += 1

            if step % 100 == 0:
                rec = dict(step=step, loss=loss.item(), lr=lr_at(step, cfg),
                           sec_per_step=(time.time() - t0) / 100)
                t0 = time.time()
                if step % cfg.eval_every == 0:
                    rec["val_loss"] = evaluate(ema.module, val_loader, device)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                ema.module.eval()
                grids = sample(ema.module, 16, H * W, cfg.sample_steps, cfg.remask,
                               cfg.sample_temperature)
                text = "\n\n".join(render(g, H, W) for g in grids)
                (run / f"samples_{step:06d}.txt").write_text(text)
                torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(),
                                opt=opt.state_dict(), step=step, config=asdict(cfg)),
                           ckpt_path)
            if step >= cfg.max_steps:
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="override config fields, key=value")
    args = ap.parse_args()
    main(load_config(args.config, args.set))
