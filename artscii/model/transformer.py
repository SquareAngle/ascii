"""Bidirectional transformer mask predictor (LLaDA style: no timestep input — the
fraction of MASK tokens in the input already tells the model the noise level)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .rope2d import apply_rope, rope_2d

# name -> (d_model, n_layers, n_heads); used for the scaling sweep.
PRESETS = {
    "tiny": (256, 6, 4),
    "small": (384, 8, 6),
    "base": (512, 12, 8),
    "large": (768, 12, 12),
    "xl": (1024, 16, 16),
}


class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = 64 * round(8 * d / 3 / 64)
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn, self.mlp = Attention(d, heads, dropout), SwiGLU(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        return x + self.drop(self.mlp(self.norm2(x)))


class MaskPredictor(nn.Module):
    def __init__(self, vocab_size, H, W, size="small", dropout=0.0, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt  # recompute activations in backward: less memory, ~30% slower
        d, layers, heads = PRESETS[size]
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        cos, sin = rope_2d(H, W, d // heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, cond=None):
        """tokens: [B, H*W] -> logits [B, H*W, vocab]. `cond` is reserved for text
        conditioning (None = unconditional, which is also the CFG null branch)."""
        x = self.embed(tokens)
        for block in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(block, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = block(x, self.cos, self.sin)
        return self.head(self.norm(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
