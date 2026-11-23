"""ASCII art as a fixed H x W grid of tokens.

Vocabulary: the 95 printable ASCII chars (space = 0), plus PAD for cells outside the
piece's bounding box and MASK for diffusion. Keeping PAD separate from space lets
the model learn where a drawing ends instead of seeing an ocean of identical blanks.
"""
import json
import random
import re

import torch
from torch.utils.data import Dataset

CHARS = [chr(i) for i in range(32, 127)]
PAD, MASK = len(CHARS), len(CHARS) + 1
VOCAB_SIZE = len(CHARS) + 2

# Horizontal flip must also mirror direction-carrying characters.
_PAIRS = ["/\\", "()", "<>", "[]", "{}", "db", "pq"]
MIRROR = {a: b for p in _PAIRS for a, b in (p, p[::-1])}
_WORD = re.compile(r"[A-Za-z]{3,}")  # pieces with words/signatures aren't flipped


def flip(lines: list[str]) -> list[str]:
    w = max(len(l) for l in lines)
    return ["".join(MIRROR.get(c, c) for c in l.ljust(w)[::-1]).rstrip() for l in lines]


def encode(lines: list[str], H: int, W: int, top: int = 0, left: int = 0) -> torch.Tensor:
    """Place the piece at (top, left); PAD everywhere outside its bounding box."""
    h, w = len(lines), max(len(l) for l in lines)
    grid = torch.full((H, W), PAD, dtype=torch.long)
    grid[top:top + h, left:left + w] = 0  # inside the box: space by default
    for r, line in enumerate(lines):
        grid[top + r, left:left + len(line)] = torch.tensor([ord(c) - 32 for c in line])
    return grid.flatten()


def render(tokens: torch.Tensor, H: int, W: int) -> str:
    """Tokens -> printable text. PAD shows as space, MASK as a shaded block."""
    rows = []
    for row in tokens.view(H, W).tolist():
        rows.append("".join(" " if t == PAD else "▒" if t == MASK else CHARS[t]
                            for t in row).rstrip())
    return "\n".join(rows).rstrip("\n")


class AsciiArtDataset(Dataset):
    """Pieces that fit the canvas, with random flip/shift augmentation.

    Returns {tokens: LongTensor[H*W], description: str} — `description` is the
    (future) text-conditioning slot; for now it is the source category.
    """

    def __init__(self, path, H, W, flip_p=0.0, shift_p=0.0, limit=None):
        self.H, self.W, self.flip_p, self.shift_p = H, W, flip_p, shift_p
        rows = [json.loads(l) for l in open(path)]
        self.items = [r for r in rows if r["h"] <= H and r["w"] <= W][:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        lines = item["art"].split("\n")
        if random.random() < self.flip_p and not _WORD.search(item["art"]):
            lines = flip(lines)
        top = left = 0
        if random.random() < self.shift_p:
            top = random.randint(0, self.H - item["h"])
            left = random.randint(0, self.W - item["w"])
        return dict(tokens=encode(lines, self.H, self.W, top, left),
                    description=item["description"])


def collate(batch):
    return dict(tokens=torch.stack([b["tokens"] for b in batch]),
                description=[b["description"] for b in batch])
