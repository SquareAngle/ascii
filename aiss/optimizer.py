"""Single-vertex simulated annealing with incremental raster and objective evaluation."""

from dataclasses import dataclass
import math
import time

import numpy as np

from .deformation import Deformation
from .geometry import rasterize, segment_cells, segment_pixels


@dataclass(frozen=True)
class SegmentFootprint:
    pixels: dict[int, list[int]]
    lengths: dict[int, float]

    @property
    def cells(self):
        return self.pixels.keys() | self.lengths.keys()


def footprint(start, end, grid):
    cells, indices = segment_pixels(start, end, grid.cell_width, grid.cell_height, grid.columns)
    pixels = {}
    for cell, index in zip(cells.tolist(), indices.tolist()):
        pixels.setdefault(cell, []).append(index)
    lengths = dict(segment_cells(start, end, grid.cell_width, grid.cell_height, grid.columns))
    return SegmentFootprint(pixels, lengths)


@dataclass(frozen=True)
class Cell:
    occupied: bool = False
    character: str = " "
    matching_error: float = 0.0
    energy: float = 0.0


EMPTY = Cell()


def score_cell(character, matching_error, lengths, costs):
    denominator = sum(lengths)
    deformation = (
        math.fsum((length / denominator) * cost for length, cost in zip(lengths, costs))
        if denominator else 1.0
    )
    return Cell(True, character, matching_error, matching_error * deformation)


class Objective:
    def __init__(self, drawing, grid, bank, config):
        self.drawing, self.grid, self.bank = drawing, grid, bank
        self.deformation = Deformation.from_drawing(drawing, config)
        self.vertices = drawing.vertices.copy()
        self.costs = self.deformation.costs(self.vertices)
        self.footprints = [footprint(self.vertices[a], self.vertices[b], grid) for a, b in drawing.edges]
        self.incident = [set() for _ in self.vertices]
        self.members = [set() for _ in range(grid.columns * grid.rows)]
        for edge, (a, b) in enumerate(drawing.edges):
            self.incident[a].add(edge)
            self.incident[b].add(edge)
            for cell in self.footprints[edge].cells:
                self.members[cell].add(edge)
        self.affected = [self.deformation.affected(tuple(edges)) for edges in self.incident]
        self.cells = [self._cell(sorted(edges), cell, {}, {}) for cell, edges in enumerate(self.members)]
        self.occupied = sum(cell.occupied for cell in self.cells)
        self.total = math.fsum(cell.energy for cell in self.cells)

    @property
    def energy(self):
        return self.total / self.occupied if self.occupied else 0.0

    def text(self):
        return text_from_cells(self.cells, self.grid)

    def _cell(self, edges, cell, footprints, costs, cached=None):
        if cached is None:
            mask = np.zeros(self.grid.cell_width * self.grid.cell_height, dtype=bool)
            for edge in edges:
                points = footprints.get(edge, self.footprints[edge]).pixels.get(cell)
                if points:
                    mask[points] = True
            if not mask.any():
                return EMPTY
            character, error = self.bank.match(mask)
        else:
            if not cached.occupied:
                return EMPTY
            character, error = cached.character, cached.matching_error

        lengths, segment_costs = [], []
        for edge in edges:
            length = footprints.get(edge, self.footprints[edge]).lengths.get(cell, 0.0)
            if length:
                lengths.append(length)
                segment_costs.append(costs.get(edge, self.costs[edge]))
        return score_cell(character, error, lengths, segment_costs)

    def propose(self, vertex, position):
        previous = self.vertices[vertex].copy()
        self.vertices[vertex] = position
        try:
            affected = self.affected[vertex]
            updated_costs = self.deformation.costs(self.vertices, affected)
            if not np.isfinite(updated_costs).all():
                return None
            costs = dict(zip(affected, updated_costs.tolist()))
            footprints, geometry_changed = {}, set()
            for edge in self.incident[vertex]:
                a, b = self.drawing.edges[edge]
                updated = footprint(self.vertices[a], self.vertices[b], self.grid)
                footprints[edge] = updated
                geometry_changed.update(updated.cells | self.footprints[edge].cells)
        finally:
            self.vertices[vertex] = previous

        members = {cell: self.members[cell].copy() for cell in geometry_changed}
        for edge, updated in footprints.items():
            for cell in self.footprints[edge].cells:
                members[cell].discard(edge)
            for cell in updated.cells:
                members[cell].add(edge)
        affected_cells = geometry_changed.copy()
        for edge in affected:
            affected_cells.update(self.footprints[edge].cells)
        cells = {}
        for cell in sorted(affected_cells):
            edges = sorted(members.get(cell, self.members[cell]))
            cached = None if cell in geometry_changed else self.cells[cell]
            cells[cell] = self._cell(edges, cell, footprints, costs, cached)
        occupied = self.occupied + sum(cell.occupied - self.cells[i].occupied for i, cell in cells.items())
        # Summing cached and updated terms avoids cancellation when a rejected or
        # exploratory state gives one segment a very large exponential penalty.
        try:
            total = math.fsum(cells.get(i, cell).energy for i, cell in enumerate(self.cells))
        except OverflowError:
            return None
        if not math.isfinite(total):
            return None
        return Proposal(vertex, position, footprints, costs, members, cells, total, occupied)

    def commit(self, proposal):
        self.vertices[proposal.vertex] = proposal.position
        for edge, updated in proposal.footprints.items():
            self.footprints[edge] = updated
        for edge, cost in proposal.costs.items():
            self.costs[edge] = cost
        for cell, edges in proposal.members.items():
            self.members[cell] = edges
        for cell, updated in proposal.cells.items():
            self.cells[cell] = updated
        self.occupied = proposal.occupied
        self.total = proposal.total

    def full(self, vertices=None):
        """Reference path using a complete raster and fresh per-segment geometry."""
        vertices = self.vertices if vertices is None else vertices
        mask = rasterize(self.drawing, vertices)
        costs = self.deformation.costs(vertices)
        pairs = [[] for _ in self.cells]
        grid = self.grid
        for edge, (a, b) in enumerate(self.drawing.edges):
            for cell, length in segment_cells(vertices[a], vertices[b], grid.cell_width,
                                              grid.cell_height, grid.columns).items():
                pairs[cell].append((length, costs[edge]))
        cells = []
        for y in range(0, grid.height, grid.cell_height):
            for x in range(0, grid.width, grid.cell_width):
                tile = mask[y:y + grid.cell_height, x:x + grid.cell_width]
                lengths, weights = (zip(*pairs[len(cells)]) if pairs[len(cells)] else ((), ()))
                cells.append(score_cell(*self.bank.match(tile), lengths, weights) if tile.any() else EMPTY)
        count = sum(cell.occupied for cell in cells)
        energy = math.fsum(cell.energy for cell in cells) / count if count else 0.0
        return energy, text_from_cells(cells, grid)


@dataclass(frozen=True)
class Proposal:
    vertex: int
    position: np.ndarray
    footprints: dict[int, SegmentFootprint]
    costs: dict[int, float]
    members: dict[int, set[int]]
    cells: dict[int, Cell]
    total: float
    occupied: int

    @property
    def energy(self):
        return self.total / self.occupied if self.occupied else 0.0


def text_from_cells(cells, grid):
    characters = "".join(cell.character for cell in cells)
    return "".join(characters[x:x + grid.columns] + "\n" for x in range(0, len(cells), grid.columns))


def temperature(initial_mean_error, iteration, mode):
    if mode == "geometric":
        return 0.2 * initial_mean_error * 0.997 ** (iteration - 1)
    power = 0.997 if mode == "printed" else -0.997
    return 0.2 * initial_mean_error * iteration ** power


def accept(delta, heat, uniform, mode):
    if delta < 0:
        return True
    probability = 1.0 if delta == 0 else math.exp(-delta / heat) if heat > 0 else 0.0
    return probability < uniform if mode == "printed" else uniform < probability


def lattice_offsets(radius):
    return np.array([(x, y) for y in range(-radius, radius + 1) for x in range(-radius, radius + 1)
                     if 0 < x * x + y * y <= radius * radius])


@dataclass(frozen=True)
class Checkpoint:
    iteration: int
    accepted: int
    current: float
    best: float


@dataclass(frozen=True)
class Optimization:
    initial_text: str
    text: str
    vertices: np.ndarray
    history: tuple[Checkpoint, ...]
    stopping_reason: str
    best_iteration: int
    runtime: float


def optimize(drawing, grid, bank, config):
    started = time.perf_counter()
    state = Objective(drawing, grid, bank, config)
    initial_text = best_text = state.text()
    best_energy, best_vertices = state.energy, state.vertices.copy()
    history = [Checkpoint(0, 0, best_energy, best_energy)]
    best_iteration = iterations = accepted = 0
    initial_mean = math.fsum(cell.matching_error for cell in state.cells) / len(state.cells)
    rng = np.random.default_rng(config.seed)
    radius = max(grid.cell_width, grid.cell_height)
    offsets = lattice_offsets(radius)

    def displacement():
        if config.proposal == "uniform_radius":
            length, angle = rng.uniform(0, radius), rng.uniform(0, 2 * math.pi)
            return length * np.array([math.cos(angle), math.sin(angle)])
        return offsets[rng.integers(len(offsets))]

    if not len(state.vertices):
        reason = "empty"
    else:
        while True:
            if iterations - best_iteration >= config.patience:
                reason = "patience"
                break
            if config.max_proposals is not None and iterations >= config.max_proposals:
                reason = "budget"
                break
            iterations += 1
            vertex = int(rng.integers(len(state.vertices)))
            position = state.vertices[vertex] + displacement()
            uniform = rng.random()
            if 0 <= position[0] <= grid.width - 1 and 0 <= position[1] <= grid.height - 1:
                proposal = state.propose(vertex, position)
                heat = temperature(initial_mean, iterations, config.temperature)
                if proposal is not None and accept(proposal.energy - state.energy, heat, uniform, config.acceptance):
                    state.commit(proposal)
                    accepted += 1
                    if state.energy < best_energy - 1e-12:
                        best_energy, best_vertices = state.energy, state.vertices.copy()
                        best_text, best_iteration = state.text(), iterations
            if iterations % 100 == 0:
                history.append(Checkpoint(iterations, accepted, state.energy, best_energy))
    if history[-1].iteration != iterations:
        history.append(Checkpoint(iterations, accepted, state.energy, best_energy))
    return Optimization(initial_text, best_text, best_vertices, tuple(history), reason, best_iteration,
                        time.perf_counter() - started)
