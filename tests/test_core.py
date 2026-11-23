"""Sanity checks for the data encoding, augmentation, 2D RoPE and masking.

    python -m tests.test_core
"""
import torch

from artscii.data.dataset import VOCAB_SIZE, encode, flip, render
from artscii.diffusion.llada import mask_tokens, sample
from artscii.model.rope2d import apply_rope, rope_2d
from artscii.model.transformer import MaskPredictor

ART = [" /\\_/\\", "( o.o )", " > ^ <"]


def test_encode_render_roundtrip():
    assert render(encode(ART, 8, 16), 8, 16) == "\n".join(ART)
    shifted = render(encode(ART, 8, 16, top=2, left=3), 8, 16).split("\n")
    assert [l[3:] for l in shifted[2:]] == ART


def test_flip_is_involution():
    assert flip(flip(ART)) == [l.rstrip() for l in ART]
    assert flip(["/("]) == [")\\"]


def test_rope_is_relative():
    H, W, d = 6, 7, 16
    cos, sin = rope_2d(H, W, d)
    q, k = torch.randn(d), torch.randn(d)
    idx = lambda r, c: r * W + c

    def score(a, b):
        qa = apply_rope(q.view(1, 1, 1, d), cos[a], sin[a]).flatten()
        kb = apply_rope(k.view(1, 1, 1, d), cos[b], sin[b]).flatten()
        return qa @ kb

    s1 = score(idx(1, 2), idx(3, 5))
    s2 = score(idx(2, 3), idx(4, 6))  # same (row, col) offset, shifted
    assert torch.allclose(s1, s2, atol=1e-4)


def test_mask_rate_matches_t():
    x = torch.zeros(2000, 512, dtype=torch.long)
    _, masked, t = mask_tokens(x)
    assert (masked.float().mean(1) - t).abs().max() < 0.1


def test_sample_fills_every_cell():
    model = MaskPredictor(VOCAB_SIZE, 4, 8, "tiny").eval()
    for remask in ("random", "low_confidence"):
        out = sample(model, 2, 32, steps=5, remask=remask)
        assert (out < VOCAB_SIZE - 1).all()  # no MASK left


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
