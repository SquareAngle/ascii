from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .config import ConversionConfig
from .geometry import Drawing, Grid, fit_to_grid
from .glyphs import glyph_bank, render_text
from .optimizer import Optimization, optimize
from .preprocess import RasterSketch, preprocess_sketch


@dataclass(frozen=True)
class ConversionResult:
    grid: Grid
    drawing: Drawing
    optimization: Optimization
    config: ConversionConfig
    runtime: float
    raster: RasterSketch | None = None

    @property
    def text(self):
        return self.optimization.text

    @property
    def initial_text(self):
        return self.optimization.initial_text

    @property
    def initial_score(self):
        return self.optimization.history[0].best

    @property
    def final_score(self):
        return self.optimization.history[-1].best

    @property
    def stopping_reason(self):
        return self.optimization.stopping_reason

    @property
    def dimensions(self):
        return self.grid.columns, self.grid.rows

    def render(self, *, initial=False, scale=1):
        return render_text(self.initial_text if initial else self.text, self.config.font, scale)

    def metadata(self):
        config = asdict(self.config)
        font_path = Path(self.config.font.path)
        config["font"]["path"] = font_path.name
        config["font"]["sha256"] = hashlib.sha256(font_path.read_bytes()).hexdigest()
        last = self.optimization.history[-1]
        return {
            "config": config,
            "columns": self.grid.columns,
            "rows": self.grid.rows,
            "vertices": len(self.drawing.vertices),
            "segments": len(self.drawing.edges),
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "iterations": last.iteration,
            "accepted": last.accepted,
            "best_iteration": self.optimization.best_iteration,
            "stopping_reason": self.stopping_reason,
            "runtime_seconds": self.runtime,
            "optimization_seconds": self.optimization.runtime,
            "history": [asdict(step) for step in self.optimization.history],
        }


def _convert(polylines, size, config, started, raster=None):
    font = config.font
    drawing, grid = fit_to_grid(polylines, size, config.columns, font.cell_width, font.cell_height,
                                config.padding, config.max_segment_length)
    bank = glyph_bank(font, config.gaussian_sigma)
    optimized = optimize(drawing, grid, bank, config)
    return ConversionResult(grid, drawing, optimized, config, time.perf_counter() - started, raster)


def convert_polylines(polylines, *, size=None, config=ConversionConfig()) -> ConversionResult:
    """Convert an iterable of [x, y] polylines. Use size=(width, height) to keep margins."""
    return _convert(polylines, size, config, time.perf_counter())


def convert_sketch(source: str | Path | Image.Image, *, config=ConversionConfig()) -> ConversionResult:
    """Convert a clean raster sketch. Raster centerlines are exposed on result.raster."""
    started = time.perf_counter()
    sketch = preprocess_sketch(source, config.raster)
    return _convert(sketch.polylines, sketch.original.size, config, started, sketch)


def convert_file(path: str | Path, *, config=ConversionConfig()) -> ConversionResult:
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return convert_polylines(data["polylines"], size=data.get("size"), config=config)
    return convert_sketch(source, config=config)
