"""Alignment-insensitive shape similarity from Section 4 and Equation 1."""

from dataclasses import dataclass

from numba import njit
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Shape:
    histogram: np.ndarray
    ink: float


class AISS:
    def __init__(self, width: int, height: int, sigma: float):
        self.width, self.height, self.sigma = width, height, sigma
        self.radius = min(width, height) / 2
        self.anchors = tuple((x, y) for y in range(1, height, 2) for x in range(1, width, 2))
        # The central disk begins at zero. Its outer edge and the next four
        # outer edges are equally spaced in log radius, starting at one pixel.
        self.radial_edges = np.geomspace(1, self.radius, 5)
        yy, xx = np.indices((height, width))
        bins, pixels = [], []
        for anchor, (x, y) in enumerate(self.anchors):
            dx, dy = xx - x, yy - y
            radius = np.hypot(dx, dy)
            valid = radius <= self.radius
            rings = np.minimum(4, np.searchsorted(self.radial_edges, radius[valid], side="right"))
            angle = np.arctan2(dy[valid], dx[valid]) % (2 * np.pi)
            sectors = np.floor(angle * (12 / (2 * np.pi))).astype(int) % 12
            bins.extend(anchor * 60 + rings * 12 + sectors)
            pixels.extend(np.flatnonzero(valid))
        self.operator = csr_matrix((np.ones(len(pixels)), (bins, pixels)),
                                   shape=(len(self.anchors) * 60, width * height))
        active = np.diff(self.operator.indptr) != 0
        self.offsets = np.r_[0, np.cumsum(active.reshape(-1, 60).sum(axis=1))]
        self.compact = self.operator[active]

    def _blur(self, image: np.ndarray) -> np.ndarray:
        return gaussian_filter(image.reshape(self.height, self.width).astype(float), self.sigma,
                               radius=3, mode="constant").ravel()

    def describe(self, image: np.ndarray) -> Shape:
        return Shape((self.operator @ self._blur(image)).reshape(-1, 60), float(image.sum()))


def distance(first: Shape, second: Shape) -> float:
    mass = first.ink + second.ink
    if mass == 0:
        return 0.0
    return float(np.linalg.norm(first.histogram - second.histogram, axis=1).sum() / mass)


@njit(cache=True)
def nearest(query, ink, glyphs, glyph_ink, offsets):
    best_index, best_score = 0, np.inf
    for glyph in range(len(glyphs)):
        mass = ink + glyph_ink[glyph]
        cutoff = best_score * mass
        total = 0.0
        for anchor in range(len(offsets) - 1):
            squared = 0.0
            for index in range(offsets[anchor], offsets[anchor + 1]):
                delta = query[index] - glyphs[glyph, index]
                squared += delta * delta
            total += np.sqrt(squared)
            if total > cutoff:
                break
        score = total / mass
        if score < best_score:
            best_index, best_score = glyph, score
    return best_index, best_score
