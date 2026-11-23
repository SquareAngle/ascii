"""Small local process pool with stable per-path seeds and independent item results."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from hashlib import blake2b
from pathlib import Path
from typing import Iterable

from .api import ConversionResult, convert_file
from .config import ConversionConfig


@dataclass(frozen=True)
class BatchItem:
    path: Path
    seed: int
    result: ConversionResult | None
    error: str | None


def _convert_item(job) -> BatchItem:
    path, config = job
    try:
        return BatchItem(path, config.seed, convert_file(path, config=config), None)
    except Exception as error:
        return BatchItem(path, config.seed, None, f"{type(error).__name__}: {error}")


def convert_batch(paths: Iterable[str | Path], *, config=ConversionConfig(), workers=2) -> list[BatchItem]:
    """Return one outcome per path in input order. Seeds depend on the supplied path spelling."""
    if workers < 1:
        raise ValueError("workers must be positive")
    jobs = []
    for value in paths:
        path = Path(value)
        digest = blake2b(digest_size=8)
        digest.update(str(config.seed).encode("ascii") + b"\0" + str(path).encode("utf-8"))
        seed = int.from_bytes(digest.digest(), "little")
        jobs.append((path, replace(config, seed=seed)))
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        return list(executor.map(_convert_item, jobs))
