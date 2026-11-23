"""Recreate the original diagnostic drawings and independently drawn analogues."""

import hashlib
import json
from math import cos, pi, sin
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def ellipse(x, y, rx, ry, n=24, start=0, end=2 * pi):
    points = [(round(x + rx * cos(t), 4), round(y + ry * sin(t), 4))
              for t in (start + (end - start) * i / n for i in range(n + 1))]
    return points


def main():
    specs = [
        ("diagnostics", "lines", (120, 76), [
            [(8, 20), (110, 20)], [(15, 63), (42, 36)], [(65, 35), (65, 65)],
            [(81, 36), (108, 63)],
        ], "Separated horizontal, vertical and diagonal strokes."),
        ("diagnostics", "ring", (110, 92), [ellipse(54, 46, 38, 35)], "One closed curve."),
        ("diagnostics", "kite", (124, 115), [
            [(58, 12), (104, 42), (58, 86), (18, 42), (58, 12)],
            [(18, 42), (58, 42), (104, 42)],
            [(58, 12), (58, 42), (58, 86), (76, 97), (51, 109)],
        ], "Closed contour with shared junctions and an attached tail."),
        ("diagnostics", "mug", (122, 93), [
            [(15, 24), (86, 24), (86, 63), (78, 77), (67, 82), (33, 82), (21, 77), (15, 65), (15, 24)],
            [(86, 30), (101, 30), (111, 37), (113, 47), (106, 59), (86, 64)],
            [(86, 38), (99, 38), (104, 43), (101, 50), (86, 55)],
            [(35, 16), (32, 11), (37, 6)], [(52, 16), (48, 10), (54, 4)],
        ], "Raster diagnostic with thick edges, open strokes, holes and junctions."),
        ("paper", "domed-chapel", (140, 104), [
            [(20, 51), (26, 36), (37, 23), (52, 15), (68, 12), (70, 5), (73, 12),
             (88, 15), (105, 26), (116, 38), (122, 51), (20, 51)],
            [(25, 51), (25, 94), (116, 94), (116, 51)],
            [(62, 94), (62, 69), (69, 61), (76, 61), (83, 69), (83, 94)],
            ellipse(42, 70, 6, 6, 12), ellipse(101, 70, 6, 6, 12),
        ], "Independently drawn analogue of the dome and window geometry in Figures 9–10."),
        ("paper", "steam-engine", (182, 110), [
            [(23, 82), (23, 49), (109, 49), (109, 33), (161, 33), (161, 82), (23, 82)],
            [(104, 32), (165, 32), (158, 23), (111, 23), (104, 32)],
            [(121, 39), (149, 39), (149, 58), (121, 58), (121, 39)],
            [(34, 49), (34, 34), (28, 25), (51, 25), (45, 34), (45, 49)],
            [(23, 66), (9, 82), (23, 82)], ellipse(52, 84, 15, 15, 20), ellipse(125, 84, 15, 15, 20),
            [(8, 101), (173, 101)],
        ], "Independently drawn steam-engine analogue of Figure 16. The paper result used Shift-JIS."),
    ]
    manifest = []
    for group, name, size, polylines, note in specs:
        path = ROOT / f"inputs/{group}/{name}.json"
        path.write_text(json.dumps({"size": size, "polylines": polylines}, separators=(",", ":")) + "\n")
        if name == "mug":
            scale = 3
            image = Image.new("L", tuple(scale * n for n in size), 255)
            draw = ImageDraw.Draw(image)
            for line in polylines:
                draw.line([(int(x * scale), int(y * scale)) for x, y in line], fill=0, width=7, joint="curve")
            path = path.with_suffix(".png")
            image.save(path)
        manifest.append({
            "group": group, "id": name, "sample_id": name, "category": name,
            "source": "https://ttwong12.github.io/papers/asciiart/asciiart.html" if group == "paper" else None,
            "input": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "license": "Original project drawing", "original_caption": None, "original_prompt": None,
            "generated_description": None, "notes": note,
            "preprocessing": "Threshold, thin, trace and simplify before fitting." if name == "mug"
            else "Fit and subdivide the original polylines.",
        })
    (ROOT / "fixtures-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
