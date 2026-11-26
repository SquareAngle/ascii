"""Train the masked-diffusion model, unconditional or text-conditioned.

    python -m artscii.train --config configs/uncond_small.yaml [--set key=value ...]
    python -m artscii.train --config configs/cond_small.yaml   # captions -> conditioned

Setting `captions` (a caption jsonl from modal_caption.py / claude_batch.py) turns on
text conditioning: a frozen FLAN-T5 encoder embeds each piece's caption every step,
the model cross-attends to it, and `cond_drop` of captions are replaced by the null
condition so the same model can sample unconditionally and with classifier-free guidance.

Writes to runs/<name>/ (shared home, readable from the login node): config.yaml,
log.jsonl, samples_<step>.txt, and ckpt.pt -> a symlink to the checkpoint on /scratch.
"""
import argparse
import html
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from artscii.data.dataset import VOCAB_SIZE, AsciiArtDataset, collate, load_captions, render
from artscii.diffusion.llada import diffusion_loss, sample
from artscii.model.transformer import MaskPredictor

# Fixed prompts for the samples written during conditional training, comparable across runs.
SAMPLE_PROMPTS = [
    "a cat", "a sitting dog", "a sailboat on the water", "a house with a chimney",
    "a rose", "a fish", "a smiling face", "a car",
    "an owl perched on a branch", "a dragon", "a coffee cup", "a palm tree on an island",
    "a rabbit", "a castle", "an airplane", "a skull",
]


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
    # Text conditioning (off while `captions` is empty)
    captions: str = ""  # caption jsonl; pieces without a usable caption get the null condition
    min_confidence: float = 0.0  # drop captions below this confidence (Claude captioner only)
    text_encoder: str = "google/flan-t5-large"
    text_max_len: int = 64
    cond_drop: float = 0.1  # fraction of captions replaced by the null condition (for CFG)
    guidance: float = 2.0  # CFG strength for the samples written during training
    # Weights & Biases (live curves); off when wandb_project is empty or WANDB_API_KEY unset
    wandb_project: str = ""
    wandb_entity: str = ""  # team or user; empty = the key's default entity


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


def make_cond(model, encoder, texts, drop_p=0.0):
    """Encode captions; empty captions and a random `drop_p` fraction become the null
    condition. Returns None for an unconditional run."""
    if encoder is None:
        return None
    ctx, mask = encoder(texts)
    drop = torch.tensor([not t for t in texts], device=mask.device)
    if drop_p > 0:
        drop |= torch.rand(len(texts), device=mask.device) < drop_p
    return model.drop_cond((ctx, mask), drop)


@torch.no_grad()
def evaluate(model, loader, device, encoder=None):
    """Val loss with a fixed masking seed, so numbers are comparable across steps/runs.
    Conditional runs also report the loss with the null condition: the gap between the
    two is how much the captions help."""
    model.eval()
    totals, n = {"val_loss": 0.0, "val_loss_uncond": 0.0}, 0
    for batch in loader:
        x = batch["tokens"].to(device)
        conds = {"val_loss": make_cond(model, encoder, batch["text"])}
        if encoder is not None:
            conds["val_loss_uncond"] = None
        for name, cond in conds.items():
            gen = torch.Generator(device).manual_seed(n)  # same masks for both
            with torch.autocast("cuda", dtype=torch.float16):
                totals[name] += diffusion_loss(model, x, cond, generator=gen).item() * len(x)
        n += len(x)
    return {k: v / max(1, n) for k, v in totals.items() if encoder is not None or k == "val_loss"}


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
    captions = load_captions(cfg.captions, cfg.min_confidence) if cfg.captions else None
    encoder = None
    if captions is not None:
        from artscii.model.text import TextEncoder
        encoder = TextEncoder(cfg.text_encoder, cfg.text_max_len).to(device)
    train_set = AsciiArtDataset(f"{cfg.data_dir}/train.jsonl", H, W,
                                0 if overfit else cfg.flip_p, 0 if overfit else cfg.shift_p,
                                limit=cfg.overfit_n or None, captions=captions)
    val_set = train_set if overfit else AsciiArtDataset(f"{cfg.data_dir}/val.jsonl", H, W,
                                                        captions=captions)
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, drop_last=not overfit,
                              collate_fn=collate, num_workers=cfg.num_workers,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, 256, collate_fn=collate, num_workers=2)

    d_text = encoder.dim if encoder is not None else None
    model = MaskPredictor(VOCAB_SIZE, H, W, cfg.size, cfg.dropout, cfg.grad_ckpt,
                          d_text).to(device)
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
    if encoder is not None:
        print(f"captioned: {train_set.caption_coverage():.1%} train, "
              f"{val_set.caption_coverage():.1%} val; text encoder {cfg.text_encoder} "
              f"(frozen, d={encoder.dim})", flush=True)
    sample_cond = None
    if encoder is not None:
        with torch.no_grad():
            sample_cond = encoder(SAMPLE_PROMPTS)

    wb = init_wandb(cfg, run, model)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    while step < cfg.max_steps:
        for batch in train_loader:
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            x = batch["tokens"].to(device, non_blocking=True)
            cond = make_cond(model, encoder, batch["text"], cfg.cond_drop)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = diffusion_loss(model, x, cond, blank_weight=cfg.blank_weight)
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
                    rec.update(evaluate(ema.module, val_loader, device, encoder))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
                if wb:
                    wb.log({k: v for k, v in rec.items() if k != "step"}, step=step)
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                ema.module.eval()
                grids = sample(ema.module, len(SAMPLE_PROMPTS) if sample_cond else 16, H * W,
                               cfg.sample_steps, cfg.remask, cfg.sample_temperature,
                               cond=sample_cond, guidance=cfg.guidance)
                labels = ([f"--- {p} (guidance {cfg.guidance})\n" for p in SAMPLE_PROMPTS]
                          if sample_cond else [""] * len(grids))
                text = "\n\n".join(l + render(g, H, W) for l, g in zip(labels, grids))
                (run / f"samples_{step:06d}.txt").write_text(text)
                if wb:
                    import wandb
                    wb.log({"samples": wandb.Html(f"<pre>{html.escape(text)}</pre>")}, step=step)
                torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(),
                                opt=opt.state_dict(), step=step, config=asdict(cfg), d_text=d_text),
                           ckpt_path)
            if step >= cfg.max_steps:
                break
    if wb:
        wb.finish()


def init_wandb(cfg, run, model):
    """Start (or resume) this run's W&B run, or return None when logging is off. The W&B run
    id is kept in runs/<name>/wandb_id, so a resumed training run continues the same curves."""
    if not cfg.wandb_project:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb_project is set but WANDB_API_KEY is not; not logging to W&B", flush=True)
        return None
    import wandb

    id_file = run / "wandb_id"
    run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
    id_file.write_text(run_id)
    return wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                      name=cfg.name, id=run_id, resume="allow",
                      config=dict(asdict(cfg), params=model.num_params()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="override config fields, key=value")
    args = ap.parse_args()
    main(load_config(args.config, args.set))
