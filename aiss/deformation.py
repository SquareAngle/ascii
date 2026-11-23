"""Local segment and fixed visibility constraints from Equations 2–4."""

from dataclasses import dataclass
import math

from numba import njit
import numpy as np

EPSILON = 1e-9


@njit(cache=True)
def local_cost(original, current, length_weight, unit_identity):
    r = math.hypot(original[0], original[1])
    changed = math.hypot(current[0], current[1])
    if r <= EPSILON:
        return 1.0
    if changed <= EPSILON:
        return np.inf
    cross = original[0] * current[1] - original[1] * current[0]
    dot = original[0] * current[0] + original[1] * current[1]
    angle = math.atan2(abs(cross), dot)
    ratio = max(r, changed) / min(r, changed)
    relative = ratio - 1 if unit_identity else ratio
    return math.exp(max((8 / math.pi) * angle, length_weight * abs(r - changed), 0.5 * relative))


@njit(cache=True)
def cross(ax, ay, bx, by):
    return ax * by - ay * bx


@njit(cache=True)
def visible_segments(vertices, edges, directions):
    """Return distinct first hits, then project each midpoint onto its target."""
    capacity = len(edges) * min(len(edges), len(directions))
    sources = np.empty(capacity, dtype=np.int64)
    targets = np.empty(capacity, dtype=np.int64)
    parameters = np.empty(capacity)
    distances = np.empty(capacity)
    count = 0
    for source, (a, b) in enumerate(edges):
        px, py = (vertices[a] + vertices[b]) / 2
        visible = np.zeros(len(edges), dtype=np.bool_)
        for dx, dy in directions:
            best_distance, hit = np.inf, -1
            for target, (c, d) in enumerate(edges):
                if target == source or c == d:
                    continue
                qx = vertices[c, 0] - px
                qy = vertices[c, 1] - py
                sx = vertices[d, 0] - vertices[c, 0]
                sy = vertices[d, 1] - vertices[c, 1]
                determinant = cross(dx, dy, sx, sy)
                if abs(determinant) <= EPSILON:
                    continue
                ray_t = cross(qx, qy, sx, sy) / determinant
                segment_t = cross(qx, qy, dx, dy) / determinant
                if -EPSILON <= segment_t <= 1 + EPSILON and EPSILON < ray_t < best_distance:
                    best_distance, hit = ray_t, target
            if hit >= 0:
                visible[hit] = True
        for target in np.flatnonzero(visible):
            c, d = edges[target]
            sx, sy = vertices[d] - vertices[c]
            qx, qy = np.array([px, py]) - vertices[c]
            parameter = min(1.0, max(0.0, (qx * sx + qy * sy) / (sx * sx + sy * sy)))
            distance = math.hypot(sx * parameter - qx, sy * parameter - qy)
            if distance > EPSILON:
                sources[count] = source
                targets[count] = target
                parameters[count] = parameter
                distances[count] = distance
                count += 1
    return sources[:count], targets[:count], parameters[:count], distances[:count]


@dataclass
class Deformation:
    edges: np.ndarray
    vectors: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    parameters: np.ndarray
    weights: np.ndarray
    links: np.ndarray
    neighbor_segments: tuple[tuple[int, ...], ...]
    offsets: np.ndarray
    length_weight: float
    unit_identity: bool

    @classmethod
    def from_drawing(cls, drawing, config):
        vertices, edges = drawing.vertices, drawing.edges
        angles = np.arange(config.accessibility_rays) * (2 * np.pi / max(1, config.accessibility_rays))
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        sources, targets, parameters, distances = visible_segments(vertices, edges, directions)
        weights = 1 / distances if config.accessibility_weights == "inverse_distance" else distances
        totals = np.bincount(sources, weights=weights, minlength=len(edges))
        weights = weights / totals[sources]
        vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        links = link_vectors(vertices, edges, sources, targets, parameters)
        neighbors = [set() for _ in edges]
        for source, target in zip(sources, targets):
            neighbors[target].add(int(source))
        return cls(edges, vectors, sources, targets, parameters, weights, links,
                   tuple(tuple(sorted(items)) for items in neighbors),
                   np.searchsorted(sources, np.arange(len(edges) + 1)),
                   2 / min(config.font.cell_width, config.font.cell_height),
                   config.relative_length == "unit_identity")

    def affected(self, changed: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(set(changed).union(*(self.neighbor_segments[i] for i in changed))))

    def costs(self, vertices, affected=None):
        if affected is None:
            affected = range(len(self.edges))
        return segment_costs(vertices, self.edges, self.vectors, self.links, self.targets,
                             self.parameters, self.weights, self.offsets,
                             np.asarray(affected, dtype=np.int64), self.length_weight,
                             self.unit_identity)


def link_vectors(vertices, edges, sources, targets, parameters):
    source = edges[sources]
    target = edges[targets]
    midpoint = (vertices[source[:, 0]] + vertices[source[:, 1]]) / 2
    start, end = vertices[target[:, 0]], vertices[target[:, 1]]
    return start + parameters[:, None] * (end - start) - midpoint


@njit(cache=True)
def segment_costs(vertices, edges, original, links, targets, parameters, weights,
                  offsets, affected, length_weight, unit_identity):
    costs = np.empty(len(affected))
    for index, segment in enumerate(affected):
        a, b = edges[segment]
        start, end = vertices[a], vertices[b]
        local = local_cost(original[segment], end - start, length_weight, unit_identity)
        access = 0.0
        for link in range(offsets[segment], offsets[segment + 1]):
            c, d = edges[targets[link]]
            position = vertices[c] + parameters[link] * (vertices[d] - vertices[c])
            deformed = position - (start + end) / 2
            access += weights[link] * local_cost(links[link], deformed, length_weight, unit_identity)
        costs[index] = max(local, access)
    return costs
