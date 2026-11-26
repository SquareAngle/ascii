"""LLaDA-style masked diffusion: forward masking, training loss, iterative unmasking.

Forward process: sample t ~ U(eps, 1) and mask each cell independently with prob t.
Loss: cross-entropy on masked cells weighted by 1/t, averaged over all cells — an
upper bound on the negative log-likelihood (LLaDA eq. 3).
Sampling: start fully masked; at each of `steps` steps predict every masked cell,
commit a fixed number of them, and leave (re-mask) the rest. Which cells get
committed is the remasking strategy: random, or the most confident predictions.
Classifier-free guidance (text-conditioned models): each step also predicts with the null
condition and extrapolates, logits = uncond + w * (cond - uncond); w = 1 is plain
conditional sampling (LLaDA writes the same rule with w' = w - 1).
"""
import torch
import torch.nn.functional as F

from artscii.data.dataset import MASK, PAD


def mask_tokens(x, eps=1e-3, generator=None):
    t = eps + (1 - eps) * torch.rand(x.shape[0], device=x.device, generator=generator)
    masked = torch.rand(x.shape, device=x.device, generator=generator) < t[:, None]
    return torch.where(masked, MASK, x), masked, t


def diffusion_loss(model, x, cond=None, blank_weight=1.0, generator=None):
    """blank_weight < 1 down-weights space/PAD targets, which dominate the grid."""
    xt, masked, t = mask_tokens(x, generator=generator)
    logits = model(xt, cond)
    ce = F.cross_entropy(logits.float().transpose(1, 2), x, reduction="none")  # [B, L]
    weight = masked.float() / t[:, None]
    if blank_weight != 1.0:
        blank = (x == 0) | (x == PAD)
        weight = weight * torch.where(blank, blank_weight, 1.0)
    return (ce * weight).sum() / x.numel()


@torch.no_grad()
def sample(model, n, length, steps=128, remask="random", temperature=1.0, cond=None,
           return_trajectory=False, x_init=None, guidance=1.0):
    """Generate n grids of `length` cells. Returns tokens [n, length] (and a list of
    intermediate grids if return_trajectory). `x_init` [n, length] starts from a partially
    masked grid instead of all-MASK (inpainting); its unmasked cells are kept fixed.
    `cond` is (ctx [n, T, d_text], mask [n, T]) for a text-conditioned model, with
    classifier-free guidance strength `guidance` (1 = none)."""
    device = next(model.parameters()).device
    x = (torch.full((n, length), MASK, dtype=torch.long, device=device) if x_init is None
         else x_init.clone().to(device))
    m0 = (x == MASK).sum(1)  # cells to fill, per grid
    guided = cond is not None and guidance != 1.0
    if guided:  # one batched pass: [cond; null]
        ctx, cmask = cond
        null_ctx, null_mask = model.null_cond(n, ctx.shape[1])
        pair = (torch.cat([ctx, null_ctx.to(ctx.dtype)]), torch.cat([cmask, null_mask]))
    trajectory = []
    for step in range(steps):
        still_masked = x == MASK
        # Linear schedule: after this step, m0 * (1 - (step+1)/steps) cells stay masked.
        n_commit = still_masked.sum(1) - (m0 * (1 - (step + 1) / steps)).long()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            if guided:
                logits_c, logits_u = model(torch.cat([x, x]), pair).float().chunk(2)
                logits = logits_u + guidance * (logits_c - logits_u)
            else:
                logits = model(x, cond).float()
        logits[..., MASK] = -float("inf")
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        pred = probs.argmax(-1) if temperature == 0 else torch.multinomial(
            probs.view(-1, probs.shape[-1]), 1).view(n, length)
        if remask == "low_confidence":
            score = probs.gather(-1, pred[..., None]).squeeze(-1)
        elif remask == "random":
            score = torch.rand(n, length, device=device)
        else:
            raise ValueError(remask)
        score = score.masked_fill(~still_masked, -float("inf"))
        order = score.argsort(dim=1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(length, device=device)
                                                .expand(n, -1))
        commit = rank < n_commit[:, None]
        x = torch.where(commit, pred, x)
        if return_trajectory:
            trajectory.append(x.cpu())
    return (x, trajectory) if return_trajectory else x
