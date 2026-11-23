from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

PRINTABLE_ASCII = "".join(map(chr, range(32, 127)))
DEFAULT_FONT = Path(__file__).parent / "assets" / "LiberationMono-Regular.ttf"


@dataclass(frozen=True)
class FontConfig:
    path: Path = DEFAULT_FONT
    cell_width: int = 12
    cell_height: int = 20
    size: int = 20
    baseline: int = 15
    characters: str = PRINTABLE_ASCII

    def __post_init__(self):
        if min(self.cell_width, self.cell_height) < 4 or self.size < 1:
            raise ValueError("font cells must be at least 4 × 4 and size must be positive")
        if not 0 <= self.baseline <= self.cell_height:
            raise ValueError("baseline must be inside the cell")
        if (not self.characters.startswith(" ") or len(set(self.characters)) != len(self.characters)
                or any(c not in PRINTABLE_ASCII for c in self.characters)):
            raise ValueError("characters must be unique printable ASCII, starting with space")


@dataclass(frozen=True)
class RasterConfig:
    threshold: int = 128
    polarity: Literal["auto", "dark", "light"] = "auto"
    simplify: float = 1.0

    def __post_init__(self):
        if not 0 <= self.threshold <= 254 or self.polarity not in {"auto", "dark", "light"}:
            raise ValueError("invalid raster threshold or polarity")
        if not math.isfinite(self.simplify) or self.simplify < 0:
            raise ValueError("simplification tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class ConversionConfig:
    columns: int = 40
    seed: int = 0
    patience: int = 5_000
    max_proposals: int | None = None
    padding: float = 2.0
    max_segment_length: float | None = 24.0
    gaussian_sigma: float = 1.4
    accessibility_rays: int = 32
    relative_length: Literal["unit_identity", "printed"] = "unit_identity"
    accessibility_weights: Literal["inverse_distance", "printed"] = "inverse_distance"
    temperature: Literal["cooling", "geometric", "printed"] = "cooling"
    proposal: Literal["lattice", "uniform_radius"] = "lattice"
    acceptance: Literal["metropolis", "printed"] = "metropolis"
    font: FontConfig = FontConfig()
    raster: RasterConfig = RasterConfig()

    def __post_init__(self):
        if self.columns < 1 or self.seed < 0 or self.patience < 1 or self.accessibility_rays < 0:
            raise ValueError("invalid columns, seed, patience, or ray count")
        if self.max_proposals is not None and self.max_proposals < 0:
            raise ValueError("proposal budget must be nonnegative")
        if not 0 <= self.padding < (min(self.font.cell_width, self.font.cell_height) - 1) / 2:
            raise ValueError("padding must fit within one cell")
        if self.max_segment_length is not None and not (
            math.isfinite(self.max_segment_length) and self.max_segment_length > 0
        ):
            raise ValueError("maximum segment length must be finite and positive")
        if not math.isfinite(self.gaussian_sigma) or self.gaussian_sigma <= 0:
            raise ValueError("Gaussian sigma must be finite and positive")
        if self.relative_length not in {"unit_identity", "printed"}:
            raise ValueError("invalid relative length model")
        if self.accessibility_weights not in {"inverse_distance", "printed"}:
            raise ValueError("invalid accessibility weights")
        if self.temperature not in {"cooling", "geometric", "printed"}:
            raise ValueError("invalid temperature schedule")
        if self.proposal not in {"lattice", "uniform_radius"} or self.acceptance not in {"metropolis", "printed"}:
            raise ValueError("invalid annealing model")
