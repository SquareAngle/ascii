"""Freeze a stratified sample from the first 64 KiB of six public NDJSON files."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

CATEGORIES = ("apple", "bicycle", "cat", "fish", "house", "sailboat")
SOURCE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
ROOT = Path(__file__).resolve().parent


def select(prefix: bytes):
    # The final range line may be incomplete.
    lines = prefix.split(b"\n")[:-1]
    quotas = {True: 3, False: 1}
    for line in lines:
        data = json.loads(line)
        recognized = data["recognized"]
        if quotas[recognized]:
            quotas[recognized] -= 1
            yield line, data
        if not any(quotas.values()):
            return
    raise ValueError("the range did not contain enough samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, help="optional directory with downloaded NDJSON prefixes")
    args = parser.parse_args()
    manifest = []
    for category in CATEGORIES:
        url = f"{SOURCE}/{category}.ndjson"
        if args.prefix_dir:
            prefix = (args.prefix_dir / f"{category}.ndjson").read_bytes()
        else:
            with urlopen(Request(url, headers={"Range": "bytes=0-65535"}), timeout=30) as stream:
                prefix = stream.read(65536)
        for line, data in select(prefix):
            sample = data["key_id"]
            path = f"inputs/quickdraw/{category}-{sample}.json"
            original = line + b"\n"
            (ROOT / path).write_bytes(original)
            manifest.append({
                "group": "real", "id": f"{category}-{sample}", "sample_id": sample,
                "source": url, "input": path, "sha256": hashlib.sha256(original).hexdigest(),
                "category": category, "original_prompt": data["word"], "original_caption": None,
                "generated_description": None, "recognized": data["recognized"],
                "attribution": "The Quick, Draw! Dataset by Google, Inc. and Quick, Draw! players",
                "license": LICENSE,
                "preprocessing": "Use Google's simplified x/y stroke vectors, then fit and subdivide for AISS.",
            })
    (ROOT / "quickdraw-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
