"""Configure."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import intake_esgf
import yaml
from esmvalcore.config import CFG

if TYPE_CHECKING:
    from collections.abc import Generator

ESMVALTOOL_INTAKE_DATA_SOURCES = (
    Path(str(resources.files("benchmarking")))
    / "config_files"
    / "esmvaltool_intake_data_sources.yml"
)


def _get_esmvaltool_intake_data_sources() -> dict[str, Any]:
    """Get ESMValTool intake-esgf data sources."""
    with ESMVALTOOL_INTAKE_DATA_SOURCES.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def reset_esmvaltool_config() -> None:
    """Rest ESMValTool configuration to default values."""
    CFG.load_from_dirs([])


@contextlib.contextmanager
def setup_esmvaltool_intake_data_source(cache_dir: Path) -> Generator[None]:
    """Set up ESMValTool intake-esgf data sources."""
    intake_esgf_options = {
        "all_indices": True,
        "esg_dataroot": [],
        "local_cache": cache_dir,
    }

    with contextlib.ExitStack() as stack:
        stack.enter_context(intake_esgf.conf.set(**intake_esgf_options))
        esmvaltool_config = deepcopy(CFG)
        for project, data_config in _get_esmvaltool_intake_data_sources().items():
            esmvaltool_config["projects"][project]["data"] = data_config["data"]
        stack.enter_context(CFG.context(esmvaltool_config))
        yield
