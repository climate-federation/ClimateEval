"""Configure."""

from dataclasses import dataclass
from pathlib import Path

from esmvalcore.config import CFG


@dataclass(kw_only=True)
class Config:
    data_dir: Path = Path.home() / ".benchmarking" / "data"
    timerange: str = "*"
    min_lat: float = -90.0
    max_lat: float = 90.0
    min_lon: float = 0.0
    max_lon: float = 360.0

    def __post_init__(self) -> None:
        """Initialize class instance."""
        self.data_dir_raw = self.data_dir / "raw"
        self.data_dir_preprocessed = self.data_dir / "preprocessed"
        self.data_dir.mkdir(parents=True, exist_ok=True)


def reset_esmvaltool_config() -> None:
    """Rest ESMValTool configuration to default values."""
    CFG.load_from_dirs([])
