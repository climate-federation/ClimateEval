"""Configure."""

from esmvalcore.config import CFG


def reset_esmvaltool_config() -> None:
    """Rest ESMValTool configuration to default values."""
    CFG.load_from_dirs([])
