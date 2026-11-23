"""2D rotary position embedding for an H x W token grid.

Each head's dims are split into pairs (i, i + d/2). The first d/4 pairs rotate by the
token's row index, the next d/4 by its column index, so q.k depends only on the
relative (row, col) offset between two cells — a shifted drawing looks the same.
"""
import torch


def rope_2d(H: int, W: int, head_dim: int, base: float = 10000.0):
    """Return (cos, sin), each [H*W, head_dim], for row-major flattened positions."""
    assert head_dim % 4 == 0
    n = head_dim // 4
    freqs = base ** (-torch.arange(n, dtype=torch.float32) / n)
    rows = torch.arange(H).repeat_interleave(W).float()
    cols = torch.arange(W).repeat(H).float()
    angles = torch.cat([rows[:, None] * freqs, cols[:, None] * freqs], dim=-1)  # [L, d/2]
    angles = torch.cat([angles, angles], dim=-1)  # pair i with i + d/2
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, L, d]."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)
