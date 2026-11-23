"""Prepare centerline polylines and cache the glyph-side AISS descriptors."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from .config import FontConfig
from .descriptor import AISS, nearest
from .geometry import make_drawing, rasterize
from .preprocess import trace_skeleton


def render_glyph(font: FontConfig, character: str, scale: int = 1) -> Image.Image:
    face = ImageFont.truetype(str(font.path), font.size * scale)
    image = Image.new("L", (font.cell_width * scale, font.cell_height * scale))
    ImageDraw.Draw(image).text((0, font.baseline * scale), character, font=face, fill=255, anchor="ls")
    return image


class GlyphBank:
    def __init__(self, font: FontConfig, sigma: float):
        self.font = font
        self.aiss = AISS(font.cell_width, font.cell_height, sigma)
        supersample = 4
        vectors = []
        for character in font.characters:
            ink = np.asarray(render_glyph(font, character, supersample)) >= 128
            skeleton = skeletonize(ink)
            paths = trace_skeleton(skeleton, tolerance=1)
            vectors.append(tuple((path + 0.5) / supersample for path in paths))
        self.vectors = tuple(vectors)
        self.masks = np.stack([
            rasterize(make_drawing(paths, font.cell_width, font.cell_height))
            for paths in self.vectors
        ])
        self.inks = self.masks.sum(axis=(1, 2))
        self.histograms = np.stack([self.aiss.compact @ self.aiss._blur(mask) for mask in self.masks])
        self._match_cached = lru_cache(maxsize=16384)(self._match)

    def match(self, mask: np.ndarray) -> tuple[str, float]:
        return self._match_cached(mask.tobytes())

    def _match(self, key: bytes) -> tuple[str, float]:
        mask = np.frombuffer(key, dtype=bool)
        ink = mask.sum()
        if not ink:
            return " ", 0.0
        query = self.aiss.compact @ self.aiss._blur(mask)
        index, score = nearest(query, ink, self.histograms, self.inks, self.aiss.offsets)
        return self.font.characters[index], score


@lru_cache(maxsize=4)
def glyph_bank(font: FontConfig, sigma: float) -> GlyphBank:
    return GlyphBank(font, sigma)


def render_text(text: str, font: FontConfig, scale: int = 1) -> Image.Image:
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    rows = text.splitlines()
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("text must be a nonempty rectangular character grid")
    width, height = font.cell_width * scale, font.cell_height * scale
    image = Image.new("L", (len(rows[0]) * width, len(rows) * height), 255)
    glyphs = {c: render_glyph(font, c, scale) for c in set("".join(rows)) - {" "}}
    for y, row in enumerate(rows):
        for x, character in enumerate(row):
            if character != " ":
                image.paste(0, (x * width, y * height, (x + 1) * width, (y + 1) * height), glyphs[character])
    return image
