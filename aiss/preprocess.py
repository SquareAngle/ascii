"""Threshold clean line drawings, thin the ink, and trace the skeleton graph."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

from .config import RasterConfig
from .geometry import Polyline


@dataclass(frozen=True)
class RasterSketch:
    original: Image.Image
    grayscale: NDArray[np.uint8]
    ink: NDArray[np.bool_]
    skeleton: NDArray[np.bool_]
    polylines: tuple[Polyline, ...]
    polarity: str


def trace_skeleton(skeleton: NDArray[np.bool_], tolerance: float = 0) -> tuple[Polyline, ...]:
    points = {tuple(point) for point in np.argwhere(skeleton)}
    neighbors = {}
    for y, x in sorted(points):
        adjacent = []
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            candidate = (y + dy, x + dx)
            if candidate not in points:
                continue
            # A diagonal shortcut would add a triangle to an orthogonal bend.
            if dy and dx and ((y + dy, x) in points or (y, x + dx) in points):
                continue
            adjacent.append(candidate)
        neighbors[(y, x)] = adjacent

    visited = set()
    paths = []

    def walk(start, next_point):
        path = [start]
        while True:
            previous = path[-1]
            visited.add(tuple(sorted((previous, next_point))))
            path.append(next_point)
            adjacent = neighbors[next_point]
            if next_point == start or len(adjacent) != 2:
                return path
            next_point = adjacent[0] if adjacent[1] == previous else adjacent[1]

    for point, adjacent in neighbors.items():
        if not adjacent:
            paths.append([point])
        elif len(adjacent) != 2:
            for neighbor in adjacent:
                if tuple(sorted((point, neighbor))) not in visited:
                    paths.append(walk(point, neighbor))
    for point, adjacent in neighbors.items():
        for neighbor in adjacent:
            if tuple(sorted((point, neighbor))) not in visited:
                paths.append(walk(point, neighbor))

    result = []
    for path in paths:
        coords = np.asarray(path, dtype=float)[:, ::-1]
        if tolerance and len(coords) > 2:
            simplified = approximate_polygon(coords, tolerance)
            closed = np.array_equal(coords[0], coords[-1])
            if not closed or len(simplified) >= 4:
                coords = simplified
        result.append(coords)
    return tuple(result)


def preprocess_sketch(source: str | Path | Image.Image,
                      config: RasterConfig = RasterConfig()) -> RasterSketch:
    if isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source).convert("RGBA")
    else:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    background = "black" if config.polarity == "light" else "white"
    original = Image.alpha_composite(Image.new("RGBA", image.size, background), image).convert("RGB")
    gray = np.asarray(original.convert("L"))
    polarity = config.polarity
    if polarity == "auto":
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        polarity = "light" if np.median(border) < 128 else "dark"
    normalized = 255 - gray if polarity == "light" else gray
    ink = normalized <= config.threshold
    skeleton = skeletonize(ink)
    return RasterSketch(original, normalized, ink, skeleton,
                        trace_skeleton(skeleton, config.simplify), polarity)
