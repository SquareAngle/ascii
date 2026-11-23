from dataclasses import replace
import math

import numpy as np
from PIL import Image, ImageDraw
import pytest

from aiss import ConversionConfig, RasterConfig, preprocess_sketch
from aiss.deformation import Deformation, local_cost
from aiss.geometry import fit_to_grid, make_drawing, normalize_paths, segment_cells
from aiss.preprocess import trace_skeleton


def test_crossing_cell_lengths_and_axis_aligned_boundary():
    cells = dict(segment_cells(np.array([2.0, 2.0]), np.array([24.0, 2.0]), 10, 10, 3))
    assert cells == pytest.approx({0: 8.0, 1: 10.0, 2: 4.0})
    diagonal = dict(segment_cells(np.array([2.0, 2.0]), np.array([18.0, 18.0]), 10, 10, 3))
    assert diagonal == pytest.approx({0: math.sqrt(128), 4: math.sqrt(128)})
    seam = dict(segment_cells(np.array([10.0, 3.0]), np.array([10.0, 16.0]), 10, 10, 3))
    assert seam == pytest.approx({1: 7.0, 4: 6.0})


def test_paper_local_angle_length_and_identity():
    e = np.array([10.0, 0])
    k = 2 / 12
    assert local_cost(e, e, k, True) == 1
    assert local_cost(e, e, k, False) == pytest.approx(math.exp(0.5))
    assert local_cost(e, np.array([0.0, 10.0]), k, True) == pytest.approx(math.exp(4))
    assert local_cost(e, np.array([12.0, 0]), k, True) == pytest.approx(math.exp(2 * k))
    assert local_cost(e, np.array([1.0, 0]), k, True) == pytest.approx(math.exp(4.5))
    assert math.isinf(local_cost(e, np.zeros(2), k, True))
    assert local_cost(np.zeros(2), np.zeros(2), k, True) == 1


def test_accessibility_first_hit_occludes_farther_segment_and_tracks_movement():
    drawing = make_drawing(normalize_paths([
        [(2, 10), (12, 10)], [(0, 14), (16, 14)], [(0, 18), (16, 18)],
    ]), 30, 30)
    deformed = Deformation.from_drawing(drawing, ConversionConfig(accessibility_rays=4))
    assert list(deformed.targets[deformed.sources == 0]) == [1]
    assert deformed.parameters[deformed.sources == 0] == pytest.approx([7 / 16])
    costs = deformed.costs(drawing.vertices + np.array([5, 5]))
    np.testing.assert_allclose(costs, 1)
    moved = drawing.vertices.copy()
    moved[drawing.edges[1], 1] += 4
    assert deformed.costs(moved)[0] == pytest.approx(math.exp(4 * 2 / 12))
    assert 0 in deformed.affected((1,))


def test_accessibility_weight_order_depends_on_interpretation():
    drawing = make_drawing(normalize_paths([
        [(5, 10), (15, 10)], [(2, 14), (18, 14)], [(2, 2), (18, 2)],
    ]), 30, 30)
    config = ConversionConfig(accessibility_rays=4)
    for mode, expected in [("inverse_distance", [2 / 3, 1 / 3]), ("printed", [1 / 3, 2 / 3])]:
        deformed = Deformation.from_drawing(drawing, replace(config, accessibility_weights=mode))
        assert deformed.weights[deformed.sources == 0] == pytest.approx(expected)


def test_trace_junction_loop_gap_and_dot_with_simplification():
    skeleton = np.zeros((18, 25), dtype=bool)
    skeleton[1:7, 4] = True
    skeleton[6, 1:8] = True
    skeleton[1, 12:17] = skeleton[5, 12:17] = True
    skeleton[1:6, 12] = skeleton[1:6, 16] = True
    skeleton[8:14, 18] = skeleton[8:14, 20] = True
    skeleton[15, 23] = True
    paths = trace_skeleton(skeleton, 0.95)
    drawing = make_drawing(paths, 25, 18)
    junction = np.flatnonzero(np.all(drawing.vertices == (4, 6), axis=1))[0]
    assert np.count_nonzero(drawing.edges == junction) == 3
    assert sum(len(path) >= 4 and path[0] == path[-1] for path in drawing.paths) == 1
    assert len(drawing.paths) == 7
    assert any(len(path) == 1 for path in drawing.paths)
    assert any(np.all(path[:, 0] == 18) for path in paths)
    assert any(np.all(path[:, 0] == 20) for path in paths)


def test_raster_polarity_and_thick_stroke_becomes_one_centerline(tmp_path):
    image = Image.new("L", (40, 25), 0)
    ImageDraw.Draw(image).line((5, 12, 34, 12), fill=255, width=5)
    path = tmp_path / "light.jpg"
    image.save(path, quality=90)
    sketch = preprocess_sketch(path, RasterConfig(simplify=0))
    assert sketch.polarity == "light"
    assert len(sketch.polylines) == 1
    assert 25 <= sketch.skeleton.sum() <= 32
    assert 9 <= sketch.polylines[0][:, 1].min() <= 13
    assert sketch.ink.sum() == 150


def test_fit_keeps_shared_vertices_and_resolution_from_canvas():
    drawing, grid = fit_to_grid([[(0, 0), (12, 10)], [(12, 10), (13, 30)]],
                                (40, 40), 8, 12, 20, 2.0, 10)
    assert (grid.columns, grid.rows) == (8, 5)
    assert drawing.paths[0][-1] == drawing.paths[1][0]
    lengths = np.linalg.norm(drawing.vertices[drawing.edges[:, 0]] - drawing.vertices[drawing.edges[:, 1]], axis=1)
    assert np.max(lengths) <= 10 + 1e-10


def test_inferred_canvas_for_vertical_line_and_point():
    drawing, grid = fit_to_grid([[(0, 0), (0, 200)]], None, 12, 12, 20, 2, None)
    assert grid.rows == 8
    np.testing.assert_allclose(drawing.vertices[:, 0], (grid.width - 1) / 2)
    point, grid = fit_to_grid([[(8, 3)]], None, 1, 12, 20, 2, None)
    np.testing.assert_allclose(point.vertices[0], [(grid.width - 1) / 2, (grid.height - 1) / 2])
