"""Generate fixed evaluation runs, exact ASCII, and source-to-result panels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import label

from aiss import ConversionConfig, convert_file, convert_polylines
from aiss.config import DEFAULT_FONT

ROOT = Path(__file__).resolve().parent


def drawing_image(paths, size, scale=3, thickness=3):
    image = Image.new("L", tuple(int(n * scale) for n in size), 255)
    draw = ImageDraw.Draw(image)
    for path in paths:
        pixels = [(float(x * scale), float(y * scale)) for x, y in path]
        if len(pixels) == 1:
            draw.point(pixels[0], fill=0)
        else:
            draw.line(pixels, fill=0, width=thickness, joint="curve")
    return image


def panels(images):
    panel_w, panel_h = 266, 224
    canvas = Image.new("RGB", (panel_w * len(images), panel_h), "#f2f4f6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(DEFAULT_FONT), 15)
    for i, (title, item) in enumerate(images):
        thumb = ImageOps.contain(item.convert("RGB"), (panel_w - 20, panel_h - 40), Image.Resampling.LANCZOS)
        left = i * panel_w
        draw.text((left + 10, 9), title, fill="#162333", font=font)
        draw.rectangle((left + 6, 30, left + panel_w - 6, panel_h - 6), fill="white")
        canvas.paste(thumb, (left + (panel_w - thumb.width) // 2, 30 + (panel_h - 36 - thumb.height) // 2))
    return canvas


def save_run(job):
    entry, config, root, variant = job
    path = ROOT / entry["input"]
    if "sha256" in entry and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"source checksum mismatch for {path}")
    if entry["group"] == "real":
        data = json.loads(path.read_text())
        polylines = [list(zip(x, y)) for x, y in data["drawing"]]
        size = [max(point[i] for path in polylines for point in path) + 1 for i in range(2)]
        result = convert_polylines(polylines, size=size, config=config)
        original = drawing_image(polylines, size, scale=2, thickness=4)
    else:
        result = convert_file(path, config=config)
        if result.raster:
            original = result.raster.original
        else:
            data = json.loads(path.read_text())
            original = drawing_image(data["polylines"], data["size"], scale=3, thickness=5)

    directory = root / variant / entry["id"]
    directory.mkdir(exist_ok=True, parents=True)
    drawing, grid = result.drawing, result.grid
    size = (grid.width, grid.height)
    starting = drawing.polylines()
    deformed = drawing.polylines(result.optimization.vertices)
    panel = [
        ("Original", original),
        ("Vectorized / fitted", drawing_image(starting, size)),
        ("Initial AISS", result.render(initial=True, scale=3)),
        ("Deformed vectors", drawing_image(deformed, size)),
        ("Final ASCII", result.render(scale=3)),
    ]
    panels(panel).save(directory / "panel.png")
    (directory / "initial.txt").write_bytes(result.initial_text.encode("ascii"))
    (directory / "final.txt").write_bytes(result.text.encode("ascii"))
    data = result.metadata()
    data["id"] = entry["id"]
    data["group"] = entry["group"]
    data["variant"] = variant
    data["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    displacement = np.linalg.norm(drawing.vertices - result.optimization.vertices, axis=1)
    data["max_displacement_pixels"] = float(displacement.max(initial=0))
    if result.raster:
        raster = result.raster
        structure = np.ones((3, 3))
        data["preprocessing"] = {
            "polarity": raster.polarity, "ink_pixels": int(raster.ink.sum()),
            "skeleton_pixels": int(raster.skeleton.sum()),
            "ink_components": label(raster.ink, structure)[1],
            "skeleton_components": label(raster.skeleton, structure)[1],
            "traced_paths": len(raster.polylines),
        }
        mask = Image.fromarray((~raster.ink).astype(np.uint8) * 255)
        skeleton = Image.fromarray((~raster.skeleton).astype(np.uint8) * 255)
        paths = drawing_image(raster.polylines, raster.original.size, scale=1, thickness=1)
        panels([("Original", original), ("Threshold", mask), ("Thinned", skeleton),
                ("Traced and simplified", paths)]).save(directory / "preprocessing.png")
    (directory / "run.json").write_text(json.dumps(data, indent=2) + "\n")
    geometry = {
        "size": size, "paths": drawing.paths, "edges": drawing.edges.tolist(),
        "initial_vertices": drawing.vertices.tolist(), "final_vertices": result.optimization.vertices.tolist(),
    }
    (directory / "geometry.json").write_text(json.dumps(geometry, separators=(",", ":")) + "\n")
    return {key: value for key, value in data.items() if key not in {"history", "config"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["diagnostics", "paper", "real", "sweep", "ambiguities", "source"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--max-proposals", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    fixtures = json.loads((ROOT / "fixtures-manifest.json").read_text())
    real = json.loads((ROOT / "quickdraw-manifest.json").read_text())
    jobs = []

    def add(entry, variant, columns, seed=0, **options):
        config = ConversionConfig(columns=columns, seed=seed, max_proposals=args.max_proposals)
        jobs.append((entry, replace(config, **options), args.output, variant))

    if args.group in {"diagnostics", "paper", "real"}:
        for entry in fixtures + real:
            if entry["group"] == args.group:
                widths = {"lines": 14, "ring": 16, "kite": 17, "mug": 19, "domed-chapel": 22, "steam-engine": 26}
                add(entry, "main", widths.get(entry["id"], 22))
    elif args.group == "sweep":
        for entry in fixtures:
            if entry["id"] in {"ring", "domed-chapel"}:
                for columns in (12, 24):
                    for seed in (0, 7):
                        add(entry, f"sweep-{columns}-{seed}", columns, seed)
    elif args.group == "ambiguities":
        for entry in fixtures:
            if entry["id"] == "domed-chapel":
                for setting in ("relative_length", "accessibility_weights", "temperature", "acceptance"):
                    add(entry, f"printed-{setting.replace('_', '-')}", 22, **{setting: "printed"})
                add(entry, "local-only", 22, accessibility_rays=0)
                add(entry, "geometric-cooling", 22, temperature="geometric")
                add(entry, "uniform-radius", 22, proposal="uniform_radius")
    else:
        reference = json.loads((ROOT / "author-reference.json").read_text())
        add(reference, "author-raster", 28)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = []
        for result in executor.map(save_run, jobs):
            results.append(result)
            print(result["id"], result["variant"], result["stopping_reason"],
                  result["iterations"], round(result["runtime_seconds"], 2), flush=True)
    (args.output / f"{args.group}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
