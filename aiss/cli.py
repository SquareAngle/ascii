import argparse
import json
from pathlib import Path
import sys

from .api import convert_file
from .config import ConversionConfig, RasterConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a clean sketch or polyline JSON into ASCII art.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--columns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5000)
    parser.add_argument("--max-proposals", type=int, help="optional safety budget; 0 for initial AISS only")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--polarity", choices=["auto", "dark", "light"], default="auto")
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--text", type=Path, help="save exact ASCII text instead of writing it to stdout")
    parser.add_argument("--render", type=Path, help="save a rendering")
    parser.add_argument("--metadata", type=Path, help="save scores, history, and run configuration as JSON")
    args = parser.parse_args(argv)
    try:
        result = convert_file(args.input, config=ConversionConfig(
            columns=args.columns, seed=args.seed, patience=args.patience, max_proposals=args.max_proposals,
            raster=RasterConfig(threshold=args.threshold, polarity=args.polarity, simplify=args.simplify),
        ))
        if args.text:
            args.text.write_bytes(result.text.encode("ascii"))
        else:
            sys.stdout.write(result.text)
        if args.render:
            result.render(scale=2).save(args.render)
        if args.metadata:
            args.metadata.write_text(json.dumps(result.metadata(), indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"ascii-aiss: {error}\n")
    return 0
