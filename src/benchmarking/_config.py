"""Configure."""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING

import iris
from esmvalcore.config import CFG, Config

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def esmvaltool_only_log_errors() -> Generator[None]:
    """Set up ESMValTool to only log errors."""
    esmvalcore_logger = logging.getLogger("esmvalcore")
    esmvaltool_logger = logging.getLogger("esmvaltool")
    current_esmvalcore_loglevel = esmvalcore_logger.level
    current_esmvaltool_loglevel = esmvaltool_logger.level
    esmvalcore_logger.setLevel(logging.ERROR)
    esmvaltool_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        esmvalcore_logger.setLevel(current_esmvalcore_loglevel)
        esmvaltool_logger.setLevel(current_esmvaltool_loglevel)


@contextlib.contextmanager
def setup_esmvaltool_config_and_logging() -> Generator[None]:
    """Set up ESMValTool for usage within this project."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(use_default_esmvaltool_config())
        stack.enter_context(esmvaltool_only_log_errors())
        yield


@contextlib.contextmanager
def setup_iris_combine_policy() -> Generator[None]:
    """Set up Iris' combine policy."""
    combine_policy = {
        "equalise_cubes_kwargs": {
            "equalise_attributes": True,
            "unify_time_units": True,
        },
        "merge_concat_sequence": "mc",
        "repeat_until_unchanged": True,
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(iris.COMBINE_POLICY.context(**combine_policy))
        yield


@contextlib.contextmanager
def use_default_esmvaltool_config() -> Generator[None]:
    """Use default ESMValTool configuration."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Do not instantiate `Config` objects directly",
            category=UserWarning,
        )
        cfg = Config()
    cfg.load_from_dirs([])
    with contextlib.ExitStack() as stack:
        stack.enter_context(CFG.context(cfg))
        yield
