from dataclasses import replace

import numpy as np
from PIL import Image
import pytest

from aiss import ConversionConfig, convert_polylines, convert_sketch
from aiss.cli import main
from aiss.geometry import Grid, make_drawing, normalize_paths
from aiss.glyphs import glyph_bank
from aiss.optimizer import Objective, accept, score_cell, temperature

PATHS = [[(2, 5), (16, 8), (23, 14)], [(4, 27), (14, 25), (21, 28)]]


def test_cell_weights_use_lengths_in_the_cell():
    cell = score_cell("X", 2, [6, 2], [1, 3])
    assert cell.energy == 3.0
    assert score_cell(".", 2, [], []).energy == 2


def test_full_and_incremental_agree_across_acceptance_rejection_and_cell_changes():
    config = ConversionConfig(columns=3)
    grid = Grid(3, 2, 12, 20)
    drawing = make_drawing(normalize_paths(PATHS), grid.width, grid.height)
    bank = glyph_bank(config.font, config.gaussian_sigma)
    objective = Objective(drawing, grid, bank, config)
    full_score, full_text = objective.full()
    assert objective.energy == pytest.approx(full_score)
    assert objective.text() == full_text
    rng = np.random.default_rng(42)
    for i in range(42):
        vertex = rng.integers(len(drawing.vertices))
        position = np.array([rng.integers(grid.width), rng.integers(grid.height)], dtype=float)
        previous = objective.vertices.copy()
        trial_vertices = previous.copy()
        trial_vertices[vertex] = position
        proposal = objective.propose(vertex, position)
        if proposal is None:
            continue
        full_score, full_text = objective.full(trial_vertices)
        assert proposal.energy == pytest.approx(full_score, abs=1e-9)
        if i % 3:
            objective.commit(proposal)
            assert objective.energy == pytest.approx(full_score, abs=1e-9)
            assert objective.text() == full_text
        else:
            np.testing.assert_array_equal(objective.vertices, previous)
            assert objective.energy == pytest.approx(objective.full()[0], abs=1e-9)


def test_cooling_and_uphill_acceptance_at_known_thresholds():
    assert temperature(10, 1, "cooling") == 2
    assert temperature(10, 100, "cooling") < 0.03
    assert temperature(10, 100, "printed") > 190
    assert accept(-1, 0, 0.9, "metropolis")
    assert accept(0, 0, 0.9, "metropolis")
    assert accept(1, 1, 0.3, "metropolis")
    assert not accept(1, 1, 0.4, "metropolis")
    assert not accept(1, 0, 0.0, "metropolis")
    assert accept(1, 1, 0.4, "printed")
    assert not accept(1, 1, 0.3, "printed")


def test_persistence_reproducibility_best_retention_and_budget():
    config = ConversionConfig(columns=9, patience=500, seed=7, temperature="printed")
    first = convert_polylines(PATHS, size=(26, 32), config=config)
    second = convert_polylines(PATHS, size=(26, 32), config=config)
    assert first.optimization.history == second.optimization.history
    assert first.text == second.text
    np.testing.assert_array_equal(first.optimization.vertices, second.optimization.vertices)
    assert first.stopping_reason == "patience"
    steps = first.optimization.history
    assert steps[-1].iteration == first.optimization.best_iteration + 500
    assert steps[-1].current > first.final_score
    objective = Objective(first.drawing, first.grid, glyph_bank(config.font, config.gaussian_sigma), config)
    score, text = objective.full(first.optimization.vertices)
    assert score == pytest.approx(first.final_score)
    assert text == first.text
    short = convert_polylines(PATHS, size=(26, 32), config=replace(config, max_proposals=0))
    assert short.stopping_reason == "budget"
    assert short.text == short.initial_text
    assert short.final_score == short.initial_score


def test_empty_grid_and_exact_cli_text(tmp_path):
    result = convert_sketch(Image.new("RGB", (40, 20), "white"),
                            config=ConversionConfig(columns=10))
    assert result.stopping_reason == "empty"
    assert result.initial_score == result.final_score == 0
    assert result.text == (" " * 10 + "\n") * result.grid.rows
    assert result.render(scale=2).size == (2 * result.grid.width, 2 * result.grid.height)

    fixture = tmp_path / "lines.json"
    fixture.write_text('{"size":[10,10],"polylines":[[[2,2],[8,8]]]}')
    output = tmp_path / "exact.txt"
    assert main([str(fixture), "--columns", "14", "--max-proposals", "0", "--text", str(output)]) == 0
    raw = output.read_bytes()
    assert all(len(row) == 14 for row in raw.splitlines())
    assert raw.endswith(b"\n")
    assert raw.count(b" \n") > 0
    assert all(byte in b"\n" or 32 <= byte <= 126 for byte in raw)
