import math

import numpy as np
import pytest

from aiss.config import FontConfig
from aiss.descriptor import AISS, Shape, distance
from aiss.glyphs import glyph_bank


@pytest.fixture(scope="module")
def bank():
    return glyph_bank(FontConfig(), 1.4)


def scalar_histogram(image, model):
    """Deliberately direct pixel loop for bin and Gaussian verification."""
    height, width = image.shape
    kernel = [math.exp(-i * i / (2 * model.sigma ** 2)) for i in range(-3, 4)]
    kernel = [weight / sum(kernel) for weight in kernel]
    hist = np.zeros((len(model.anchors), 60))
    for index, (anchor_x, anchor_y) in enumerate(model.anchors):
        for y, x in np.ndindex(image.shape):
            dx, dy = x - anchor_x, y - anchor_y
            radius = math.hypot(dx, dy)
            if radius > model.radius:
                continue
            ring = sum(radius >= edge for edge in model.radial_edges[:-1])
            angle = math.atan2(dy, dx) % (2 * math.pi)
            sector = int(angle * (12 / (2 * math.pi))) % 12
            for j, wy in enumerate(kernel, -3):
                for i, wx in enumerate(kernel, -3):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        hist[index, ring * 12 + sector] += image[y + j, x + i] * wx * wy
    return hist


def test_histogram_matches_explicit_pixel_and_gaussian_accumulation():
    model = AISS(8, 10, 1.4)
    image = np.zeros((10, 8))
    image[0, 0] = 1.0
    image[4, 3] = 0.25
    image[-1, -2] = 1.0
    descriptor = model.describe(image)
    assert len(model.anchors) == 20
    assert model.anchors[0] == (1, 1)
    assert model.anchors[-1] == (7, 9)
    np.testing.assert_allclose(descriptor.histogram, scalar_histogram(image, model), atol=1e-15)
    assert descriptor.ink == 2.25


def test_aiss_is_sum_of_anchor_norms_over_combined_ink():
    first = Shape(np.array([[3, 4], [0, 0]]), 2)
    second = Shape(np.array([[0, 0], [0, 3]]), 6)
    assert distance(first, second) == distance(second, first) == 1.0
    blank = Shape(np.zeros((2, 2)), 0)
    assert distance(blank, blank) == 0


@pytest.mark.parametrize("character", [" ", "-", "|", "/", "L", "O", "o", "6", "9"])
def test_exact_glyphs(bank, character):
    i = bank.font.characters.index(character)
    assert bank.match(bank.masks[i]) == (character, 0)


def test_compiled_candidate_ranking_matches_reference(bank):
    rng = np.random.default_rng(321)
    glyphs = [bank.aiss.describe(mask) for mask in bank.masks]
    for _ in range(12):
        mask = rng.uniform(size=bank.masks[0].shape) < rng.uniform(0.02, 0.35)
        shape = bank.aiss.describe(mask)
        scores = [distance(shape, glyph) for glyph in glyphs]
        match, score = bank.match(mask)
        assert match == bank.font.characters[np.argmin(scores)]
        assert score == pytest.approx(min(scores), abs=1e-12)


def test_small_shift_tolerated_and_rotation_and_scale_distinguished(bank):
    model = bank.aiss
    horizontal = np.zeros((20, 12), dtype=bool)
    horizontal[8, 2:10] = True
    shifted = np.roll(horizontal, 1, axis=0)
    shorter = np.zeros_like(horizontal)
    shorter[8, 4:8] = True
    vertical = np.zeros_like(horizontal)
    vertical[5:13, 5] = True
    h, s, smaller, v = map(model.describe, [horizontal, shifted, shorter, vertical])
    assert 0 < distance(h, s) < distance(h, v)
    assert 0 < distance(h, smaller) < distance(h, v)
