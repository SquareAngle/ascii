"""Shared coordinates, line rasterization and exact segment lengths in grid cells."""

from dataclasses import dataclass
import math

from numba import njit
import numpy as np
from numpy.typing import ArrayLike, NDArray

Polyline = NDArray[np.float64]


@dataclass(frozen=True)
class Drawing:
    vertices: NDArray[np.float64]
    edges: NDArray[np.int64]
    paths: tuple[tuple[int, ...], ...]
    width: int
    height: int

    def polylines(self, vertices=None) -> tuple[Polyline, ...]:
        coords = self.vertices if vertices is None else vertices
        return tuple(coords[list(path)] for path in self.paths)


@dataclass(frozen=True)
class Grid:
    columns: int
    rows: int
    cell_width: int
    cell_height: int

    @property
    def width(self) -> int:
        return self.columns * self.cell_width

    @property
    def height(self) -> int:
        return self.rows * self.cell_height


def normalize_paths(polylines) -> tuple[Polyline, ...]:
    paths = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float64)
        if not points.size:
            continue
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("polylines must contain finite [x, y] pairs")
        keep = np.r_[True, np.any(points[1:] != points[:-1], axis=1)]
        paths.append(points[keep])
    return tuple(paths)


def make_drawing(paths: tuple[Polyline, ...], width: int, height: int,
                 max_length: float | None = None) -> Drawing:
    vertices = {}
    edges = {}
    indexed_paths = []
    for points in paths:
        pieces = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            divisions = 1 if max_length is None else max(1, math.ceil(math.dist(start, end) / max_length))
            pieces.extend(start + (end - start) * (i / divisions) for i in range(1, divisions))
            pieces.append(end)
        path = tuple(vertices.setdefault(tuple(point), len(vertices)) for point in pieces)
        indexed_paths.append(path)
        pairs = zip(path[:-1], path[1:]) if len(path) > 1 else [(path[0], path[0])]
        for start, end in pairs:
            edges.setdefault(tuple(sorted((start, end))), None)
    return Drawing(
        np.asarray(list(vertices), dtype=np.float64).reshape(-1, 2),
        np.asarray(list(edges), dtype=np.int64).reshape(-1, 2),
        tuple(indexed_paths), width, height,
    )


def fit_to_grid(polylines, size: tuple[float, float] | None, columns: int,
                cell_width: int, cell_height: int, padding: float,
                max_length: float | None) -> tuple[Drawing, Grid]:
    paths = normalize_paths(polylines)
    if size is not None:
        span = np.asarray(size, dtype=float)
        if span.shape != (2,) or not np.isfinite(span).all() or np.any(span <= 0):
            raise ValueError("size must be a positive finite (width, height)")
        origin = np.zeros(2)
        if any(np.any((points < 0) | (points > span)) for points in paths):
            raise ValueError("polyline coordinates fall outside the canvas")
    elif paths:
        all_points = np.concatenate(paths)
        origin = all_points.min(axis=0)
        span = all_points.max(axis=0) - origin
        # Infer a centered viewbox for an axis-aligned single stroke or point.
        missing = span == 0
        span[missing] = max(1.0, span.max())
        origin[missing] -= span[missing] / 2
    else:
        origin = np.zeros(2)
        span = np.ones(2)

    image_width, image_height = span
    rows = max(1, math.ceil(image_height / ((cell_height / cell_width) * math.ceil(image_width / columns))))
    grid = Grid(columns, rows, cell_width, cell_height)
    available = np.array([grid.width - 1, grid.height - 1]) - 2 * padding
    scale = min(available / span)
    offset = (np.array([grid.width - 1, grid.height - 1]) - span * scale) / 2
    fitted = tuple((points - origin) * scale + offset for points in paths)
    return make_drawing(fitted, grid.width, grid.height, max_length), grid


@njit(cache=True)
def line_pixels(start, end):
    """One-pixel Bresenham line in global coordinates, with canonical endpoint order."""
    x, y = int(math.floor(start[0])), int(math.floor(start[1]))
    x1, y1 = int(math.floor(end[0])), int(math.floor(end[1]))
    if x > x1 or (x == x1 and y > y1):
        x, x1 = x1, x
        y, y1 = y1, y
    dx, dy = abs(x1 - x), -abs(y1 - y)
    y_step = 1 if y < y1 else -1
    error = dx + dy
    points = np.empty((max(dx, -dy) + 1, 2), dtype=np.int64)
    for index in range(len(points)):
        points[index, 0], points[index, 1] = x, y
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x += 1
        if twice_error <= dx:
            error += dx
            y += y_step
    return points


def rasterize(drawing: Drawing, vertices: ArrayLike | None = None) -> NDArray[np.bool_]:
    points = drawing.vertices if vertices is None else np.asarray(vertices)
    mask = np.zeros((drawing.height, drawing.width), dtype=bool)
    for a, b in drawing.edges:
        pixels = line_pixels(points[a], points[b])
        mask[pixels[:, 1], pixels[:, 0]] = True
    return mask


@njit(cache=True)
def segment_cells(start, end, cell_width, cell_height, columns):
    """Split at exact parametric crossings and assign each positive-length piece once."""
    dx, dy = end - start
    length = math.hypot(dx, dy)
    boundaries = [0.0, 1.0]
    for axis, step in ((0, cell_width), (1, cell_height)):
        lo, hi = min(start[axis], end[axis]), max(start[axis], end[axis])
        for boundary in range(int(lo // step) + 1, int(hi // step) + 1):
            fraction = (boundary * step - start[axis]) / (end[axis] - start[axis])
            if 0 < fraction < 1:
                boundaries.append(fraction)
    boundaries.sort()
    cells = {}
    for index in range(len(boundaries) - 1):
        lo, hi = boundaries[index], boundaries[index + 1]
        if length == 0 or hi - lo <= 1e-12:
            continue
        midpoint = (lo + hi) / 2
        cell = (int((start[1] + midpoint * dy) // cell_height) * columns
                + int((start[0] + midpoint * dx) // cell_width))
        cells[cell] = length * (hi - lo)
    return cells


@njit(cache=True)
def segment_pixels(start, end, cell_width, cell_height, columns):
    pixels = line_pixels(start, end)
    flat = np.empty(len(pixels), dtype=np.int64)
    cells = np.empty(len(pixels), dtype=np.int64)
    for i, (x, y) in enumerate(pixels):
        cells[i] = (y // cell_height) * columns + x // cell_width
        flat[i] = (y % cell_height) * cell_width + x % cell_width
    return cells, flat
