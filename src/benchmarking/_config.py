"""Configure."""

from dataclasses import dataclass
from pathlib import Path

from esmvalcore.config import CFG


@dataclass(kw_only=True)
class Config:
    output_dir: Path = Path.cwd() / "benchmarking_output"
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
        dirs = (self.output_dir, self.data_dir)
        for _dir in dirs:
            _dir.mkdir(parents=True, exist_ok=True)


def configure_esmvaltool(_config: Config) -> None:
    """Configure ESMValTool."""
    CFG.load_from_dirs([])  # Reset ESMValTool configuration to default values
