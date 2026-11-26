"""Checks for text conditioning: cross-attention, the null condition, caption dropout,
classifier-free guidance, and caption loading. No pretrained weights or GPU needed.

    python -m tests.test_cond
"""
import json
import tempfile

import torch

from artscii.data.dataset import VOCAB_SIZE, AsciiArtDataset, MASK, collate, load_captions
from artscii.diffusion.llada import diffusion_loss, sample
from artscii.model.transformer import MaskPredictor
from artscii.train import make_cond

H, W, DT = 4, 8, 32


def cond_model(seed=0):
    torch.manual_seed(seed)
    return MaskPredictor(VOCAB_SIZE, H, W, "tiny", d_text=DT).eval()


def rand_cond(B, T=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 3:] = False  # first caption is shorter: padded
    return torch.randn(B, T, DT, generator=g), mask


def perturb_xattn(model):
    """Zero-initialised cross-attention ignores the text; give it weights to test with."""
    for b in model.blocks:
        torch.nn.init.normal_(b.xattn.out.weight, std=0.05)


def test_zero_init_ignores_text_at_start():
    model = cond_model()
    x = torch.randint(0, VOCAB_SIZE - 1, (2, H * W))
    with torch.no_grad():
        assert torch.allclose(model(x, rand_cond(2)), model(x, None), atol=1e-6)


def test_text_changes_output_and_padding_is_ignored():
    model = cond_model()
    perturb_xattn(model)
    x = torch.randint(0, VOCAB_SIZE - 1, (2, H * W))
    ctx, mask = rand_cond(2)
    with torch.no_grad():
        out = model(x, (ctx, mask))
        assert not torch.allclose(out, model(x, None), atol=1e-4)
        ctx2 = ctx.clone()
        ctx2[0, 3:] = 100.0  # garbage in padded positions must not matter
        assert torch.allclose(model(x, (ctx2, mask)), out, atol=1e-5)


def test_drop_cond_equals_null():
    model = cond_model()
    perturb_xattn(model)
    x = torch.randint(0, VOCAB_SIZE - 1, (2, H * W))
    ctx, mask = rand_cond(2)
    dropped = model.drop_cond((ctx, mask), torch.tensor([True, False]))
    with torch.no_grad():
        out = model(x, dropped)
        assert torch.allclose(out[0], model(x[:1], None)[0], atol=1e-5)
        assert torch.allclose(out[1], model(x, (ctx, mask))[1], atol=1e-5)


def test_null_token_gets_gradient():
    model = cond_model().train()
    perturb_xattn(model)
    x = torch.randint(0, VOCAB_SIZE - 2, (4, H * W))
    cond = model.drop_cond(rand_cond(4), torch.tensor([True, False, True, False]))
    diffusion_loss(model, x, cond).backward()
    assert model.null.grad is not None and model.null.grad.abs().sum() > 0
    assert model.blocks[0].xattn.kv.weight.grad.abs().sum() > 0


def test_guided_sampling_fills_and_w1_matches_unguided():
    model = cond_model()
    perturb_xattn(model)
    cond = rand_cond(2)
    for w in (1.0, 3.0):
        torch.manual_seed(0)
        x = sample(model, 2, H * W, steps=8, cond=cond, guidance=w)
        assert (x != MASK).all()
    torch.manual_seed(0)
    a = sample(model, 2, H * W, steps=8, cond=cond, guidance=1.0)
    torch.manual_seed(0)
    b = sample(model, 2, H * W, steps=8, cond=cond)
    assert torch.equal(a, b)


def test_unconditional_model_unchanged():
    model = MaskPredictor(VOCAB_SIZE, H, W, "tiny").eval()
    assert not hasattr(model.blocks[0], "xattn") and not hasattr(model, "null")
    x = sample(model, 2, H * W, steps=4)
    assert (x != MASK).all()


class StubEncoder:
    """Stands in for TextEncoder: a deterministic embedding per text, one token per word."""
    dim = DT

    def __call__(self, texts):
        T = max(1, max(len(t.split()) for t in texts))
        ctx = torch.zeros(len(texts), T, DT)
        mask = torch.zeros(len(texts), T, dtype=torch.bool)
        for i, t in enumerate(texts):
            for j, w in enumerate(t.split() or [""]):
                ctx[i, j] = torch.randn(DT, generator=torch.Generator().manual_seed(hash(w) % 2**31))
                mask[i, j] = True
        return ctx, mask


def test_make_cond_nulls_empty_captions():
    model = cond_model()
    ctx, mask = make_cond(model, StubEncoder(), ["a cat", "", "a dog on a mat"], drop_p=0.0)
    assert ctx.shape == (3, 5, DT)
    assert torch.equal(ctx[1, 0], model.null) and mask[1].tolist() == [True, False, False, False, False]
    assert not torch.equal(ctx[0, 0], model.null)
    torch.manual_seed(0)
    ctx, mask = make_cond(model, StubEncoder(), ["a cat"] * 2000, drop_p=0.1)
    frac = (ctx[:, 0] == model.null).all(-1).float().mean().item()
    assert 0.07 < frac < 0.13, frac
    assert make_cond(model, None, ["a cat"]) is None


def test_captions_join_dataset():
    from artscii.caption.vlm import piece_key

    arts = [" /\\_/\\\n( o.o )\n > ^ <", "|--|\n|  |\n|--|", "~~~~~~\n ~~~~ "]
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/train.jsonl", "w") as f:
            for a in arts:
                f.write(json.dumps(dict(art=a, h=len(a.split("\n")),
                                        w=max(map(len, a.split("\n"))), description="x")) + "\n")
        rows = [dict(key=piece_key(arts[0]), short="a cat", medium="a cat face",
                     long="a cat face with whiskers", confidence=0.95),
                dict(key=piece_key(arts[1]), short="a box", medium="b", long="c", confidence=0.3),
                dict(key=piece_key(arts[2]), unclear=True)]
        with open(f"{d}/caps.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        caps = load_captions(f"{d}/caps.jsonl", min_confidence=0.5)
        assert set(caps) == {piece_key(arts[0])}
        assert len(load_captions(f"{d}/caps.jsonl")) == 2  # unclear always dropped
        ds = AsciiArtDataset(f"{d}/train.jsonl", H * 2, W * 2, captions=caps)
        assert ds.caption_coverage() == 1 / 3
        texts = collate([ds[i] for i in range(3)])["text"]
        assert texts[0] in caps[piece_key(arts[0])] and texts[1] == texts[2] == ""


def test_param_count_added():
    base = MaskPredictor(VOCAB_SIZE, H, W, "small").num_params()
    cond = MaskPredictor(VOCAB_SIZE, H, W, "small", d_text=1024).num_params()
    d = 384
    assert cond - base == 8 * (2 * d * d + 2 * d * 1024 + d) + 1024


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
