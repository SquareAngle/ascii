"""Reproduce the Figure 7 placement probes and enumerate legal first moves."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from aiss import ConversionConfig
from aiss.descriptor import distance
from aiss.geometry import fit_to_grid, line_pixels
from aiss.glyphs import glyph_bank
from aiss.optimizer import Objective, lattice_offsets

ROOT = Path(__file__).resolve().parent


def placement_probes(bank):
    queries = [
        ("bottom-left corner", [((1, 1), (1, 16)), ((1, 16), (10, 16))], "L"),
        ("left vertical", [((1, 1), (1, 16))], "}"),
        ("top line with upward spur", [((1, 3), (10, 3)), ((8, 1), (8, 3))], "~"),
        ("long horizontal above center", [((1, 9), (10, 9))], "-"),
        ("long horizontal below center", [((1, 11), (10, 11))], "-"),
    ]
    glyphs = [bank.aiss.describe(mask) for mask in bank.masks]
    runs = []
    for label, lines, target in queries:
        mask = np.zeros_like(bank.masks[0])
        for start, end in lines:
            pixels = line_pixels(np.asarray(start, dtype=float), np.asarray(end, dtype=float))
            mask[pixels[:, 1], pixels[:, 0]] = True
        shape = bank.aiss.describe(mask)
        scores = sorted((distance(shape, glyph), c) for c, glyph in zip(bank.font.characters, glyphs))
        runs.append({"query": label, "pixel_lines": lines, "paper_match": target,
                     "top_five": [{"character": c, "score": s} for s, c in scores[:5]]})
    return runs


def first_moves(bank, entry, columns, rays):
    data = json.loads((ROOT / entry).read_text())
    font = bank.font
    config = replace(ConversionConfig(columns=columns), accessibility_rays=rays)
    drawing, grid = fit_to_grid(data["polylines"], data["size"], columns, font.cell_width,
                                font.cell_height, config.padding, config.max_segment_length)
    state = Objective(drawing, grid, bank, config)
    changes = []
    offsets = lattice_offsets(max(grid.cell_width, grid.cell_height))
    in_bounds = 0
    for vertex, original in enumerate(drawing.vertices):
        for offset in offsets:
            position = original + offset
            if not (0 <= position[0] <= grid.width - 1 and 0 <= position[1] <= grid.height - 1):
                continue
            in_bounds += 1
            proposal = state.propose(vertex, position)
            if proposal is not None:
                changes.append(proposal.energy - state.energy)
    lengths = np.linalg.norm(state.deformation.links, axis=1)
    return {"input": entry, "columns": columns, "rays": rays, "vertices": len(drawing.vertices),
            "links": len(lengths), "minimum_link_length": float(min(lengths)) if len(lengths) else None,
            "valid_moves": len(changes), "bounds_valid_moves": in_bounds,
            "total_moves": len(drawing.vertices) * len(offsets), "best_delta": min(changes),
            "strict_improvements": sum(delta < -1e-12 for delta in changes)}


def main():
    config = ConversionConfig()
    bank = glyph_bank(config.font, config.gaussian_sigma)
    data = {
        "placement_probes": placement_probes(bank),
        "initial_moves": [
            first_moves(bank, path, columns, rays)
            for path, columns in (("inputs/diagnostics/kite.json", 17), ("inputs/paper/domed-chapel.json", 22))
            for rays in (32, 0)
        ],
    }
    (ROOT / "results/probes.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
