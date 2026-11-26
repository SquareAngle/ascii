"""Frozen text encoder for caption conditioning: the encoder half of FLAN-T5.

Captions are encoded on the fly every training step (no cached embeddings): the encoder
sees ~64 tokens per caption while the denoiser sees 2048 grid cells, so this is a few
percent of the step's compute. It is never trained, never in the optimizer or EMA, and
never saved in checkpoints (only its name is, in the run config).
"""
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, name: str = "google/flan-t5-large", max_len: int = 64):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = T5EncoderModel.from_pretrained(name).eval().requires_grad_(False)
        self.max_len = max_len
        self.dim = self.encoder.config.d_model  # 1024 for flan-t5-large

    def to(self, device):
        # T5 overflows in fp16; bf16 is safe where the GPU supports it (A100/L40S/H100),
        # otherwise (V100, CPU) stay in fp32.
        dev = torch.device(device)
        dtype = (torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float32)
        self.encoder.to(dev, dtype=dtype)
        return self

    def train(self, mode=True):  # always eval: no dropout in the frozen encoder
        return super().train(False)

    @torch.no_grad()
    def forward(self, texts: list[str]):
        """texts -> (ctx [B, T, dim] float32, mask [B, T] bool, True = real token)."""
        device = next(self.encoder.parameters()).device
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt").to(device)
        with torch.autocast(device.type, enabled=False):
            out = self.encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask)
        return out.last_hidden_state.float(), tok.attention_mask.bool()
