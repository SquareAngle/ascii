# ASCII art generation

The project explores text-to-ASCII generation using masked diffusion. This milestone adds a contour-based converter for preparing and evaluating sketch data.

`aiss` converts clean line drawings to printable ASCII using the shape descriptor and constrained deformation in Xu, Zhang, and Wong's 2010 paper, *Structure-based ASCII Art*. It accepts PNG, JPG, and polyline JSON. Output dimensions follow the input aspect ratio and the bundled Liberation Mono font metrics.

## Install and run

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
ascii-aiss evaluation/inputs/diagnostics/mug.png --columns 19 \
  --text mug.txt --render mug-ascii.png --metadata mug-run.json
python -m pytest -q
```

Use `--max-proposals 0` for initial AISS matching. The default annealer runs until the best score has held for 5,000 proposals. An optional proposal cap reports `budget` when reached. Text uses a fixed-width rectangle with trailing spaces and one LF per row. View it using Liberation Mono for the reference placement.

## Python API

```python
from aiss import ConversionConfig, convert_polylines, convert_sketch

config = ConversionConfig(columns=40, seed=0)
result = convert_sketch("sketch.png", config=config)
print(result.text, end="")
result.render(scale=2).save("ascii.png")

vector = convert_polylines(
    [[(4, 30), (20, 5), (38, 30), (4, 30)]],
    size=(42, 35),
    config=ConversionConfig(columns=18, seed=0),
)
```

Polyline JSON uses `{"size": [42, 35], "polylines": [[[4, 30], [20, 5], [38, 30], [4, 30]]]}`. The canvas size is optional. Points are in image coordinates, with x increasing right and y increasing down. Repeat a path's first point to close it. Matching endpoints share a vertex.

`result.initial_text`, `result.dimensions`, `result.metadata()`, `result.raster`, `result.drawing`, and `result.optimization.vertices` expose the diagnostic data. `RasterConfig` sets threshold, polarity and simplification independently of matching. The API also supplies `FontConfig` for a fixed-width font with a known cell size and baseline.

After the single-image checks, use `convert_batch(paths, config=config, workers=2)` for local batches. It returns results in input order with a stable seed derived from each supplied path and the base seed, plus a per-file error when conversion fails. In a standalone script, call it under `if __name__ == "__main__":` so process startup works on macOS and Windows.

## Reproduction

[Method specification](docs/method.md) · [Dataset report](docs/datasets.md) · [Gallery and exact text](evaluation/GALLERY.md) · [Evaluation and commands](docs/evaluation.md)

The reference runs show recognizable contours on several inputs and substantial failures on other clean vectors. Published visual quality remains unverified.
