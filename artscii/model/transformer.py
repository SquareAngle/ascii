"""Bidirectional transformer mask predictor (LLaDA style: no timestep input — the
fraction of MASK tokens in the input already tells the model the noise level).

Text conditioning (d_text set): every block gets a cross-attention sublayer in which grid
cells (queries) read the caption's encoder states (keys/values). A condition is a pair
(ctx [B, T, d_text], mask [B, T] bool). The unconditional / classifier-free-guidance null
branch is a single learned "null" text token, so one network learns p(x | caption) and
p(x).
"""
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


class CrossAttention(nn.Module):
    """Grid cells attend to text tokens. The output projection is zero-initialised (see
    MaskPredictor) so a fresh layer is a no-op and conditioning is learned gradually."""

    def __init__(self, d, d_text, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d_text, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, ctx, ctx_mask):
        B, L, D = x.shape
        T, h = ctx.shape[1], self.heads
        q = self.q(x).view(B, L, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, T, 2, h, D // h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
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
    def __init__(self, d, heads, dropout, d_text=None):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn, self.mlp = Attention(d, heads, dropout), SwiGLU(d)
        if d_text:
            self.norm_x, self.xattn = nn.RMSNorm(d), CrossAttention(d, d_text, heads, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, ctx=None, ctx_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        if ctx is not None:
            x = x + self.drop(self.xattn(self.norm_x(x), ctx, ctx_mask))
        return x + self.drop(self.mlp(self.norm2(x)))


class MaskPredictor(nn.Module):
    def __init__(self, vocab_size, H, W, size="small", dropout=0.0, grad_ckpt=False,
                 d_text=None):
        super().__init__()
        self.grad_ckpt = grad_ckpt  # recompute activations in backward: less memory, ~30% slower
        self.d_text = d_text  # text encoder width; None = unconditional model
        d, layers, heads = PRESETS[size]
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout, d_text) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        cos, sin = rope_2d(H, W, d // heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        if d_text:
            self.null = nn.Parameter(torch.randn(d_text) * 0.02)  # the "no caption" token
            for block in self.blocks:
                nn.init.zeros_(block.xattn.out.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def null_cond(self, B, T=1):
        """The null condition, padded to T tokens: the null token first, the rest masked out."""
        ctx = torch.zeros(B, T, self.d_text, device=self.null.device, dtype=self.null.dtype)
        ctx[:, 0] = self.null
        mask = torch.zeros(B, T, dtype=torch.bool, device=self.null.device)
        mask[:, 0] = True
        return ctx, mask

    def drop_cond(self, cond, drop):
        """Replace the condition with the null one where drop [B] is True (caption dropout
        for classifier-free guidance, and pieces without a caption)."""
        ctx, mask = cond
        null_ctx, null_mask = self.null_cond(*mask.shape)
        return (torch.where(drop[:, None, None], null_ctx, ctx),
                torch.where(drop[:, None], null_mask, mask))

    def forward(self, tokens, cond=None):
        """tokens: [B, H*W] -> logits [B, H*W, vocab]. cond: (ctx [B, T, d_text], mask [B, T])
        or None, which for a text-conditioned model means the null (unconditional) branch."""
        ctx = mask = None
        if self.d_text:
            ctx, mask = cond if cond is not None else self.null_cond(tokens.shape[0])
        x = self.embed(tokens)
        for block in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(block, x, self.cos, self.sin, ctx, mask, use_reentrant=False)
            else:
                x = block(x, self.cos, self.sin, ctx, mask)
        return self.head(self.norm(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
