"""Provide ESMValTool CMORizer data sources."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from esmvalcore.config import CFG
from esmvaltool.cmorizers.data.cmorizer import DataCommand
from loguru import logger

from benchmarking import setup_esmvaltool_config_and_logging
from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from collections.abc import Generator

    from benchmarking import Variable


logger = logger.opt(colors=True)


class ESMValToolCMORizerDataSource(DataSource):
    """Manage ESMValTool CMORizer data source."""

    SUPPORTED_VAR_NAMES: ClassVar[tuple[str, ...] | None] = None
    """Supported variable names."""

    _FILE_PREFIX_PATTERN: ClassVar[str] = "*"
    """Prefix to only select specific files."""

    _TIER: ClassVar[int] = 2
    """Tier of data set."""

    def download(self, variable: Variable, data_root_dir: Path) -> list[Path]:
        """Download data.

        See
        https://docs.esmvaltool.org/en/latest/input.html#using-a-cmorizer-script.

        """
        self._check_supported_variables(variable)

        # Raw downloaded data by ESMValTool will be stored in temporary
        # directory. Our "raw" data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Download data
        with self._set_esmvaltool_output_dir(tmp_dir):
            DataCommand().download(self.name, original_data_dir=tmp_dir)

        # Format data
        return self._esmvaltool_cmorizer_format(
            data_root_dir,
            variable,
            download_dir=tmp_dir,
        )

    def _check_supported_variables(self, variable: Variable) -> None:
        """Raise an error if a variable is not supported."""
        if (
            self.SUPPORTED_VAR_NAMES is not None
            and variable.var_name not in self.SUPPORTED_VAR_NAMES
        ):
            msg = f"{self.id} does not provide variable '{variable.var_name}'"
            raise ValueError(msg)

    def _esmvaltool_cmorizer_format(
        self,
        data_root_dir: Path,
        variable: Variable,
        *,
        download_dir: Path,
    ) -> list[Path]:
        """Use ESMValTool CMORizer to format data.

        See
        https://docs.esmvaltool.org/en/latest/input.html#using-a-cmorizer-script.

        """
        logger.debug(f"Formatting {self.id} data (this may take while...)")

        # Store all ESMValTool output in temporary directory and only save the
        # relevant files.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with self._set_esmvaltool_output_dir(tmp_dir):
            DataCommand().format(self.name, original_data_dir=download_dir)

        latest_esmvaltool_output_dir = (
            sorted(tmp_dir.glob("data_formatting_*"))[-1]
            / f"Tier{self._TIER}"
            / self.name
        )
        raw_files = latest_esmvaltool_output_dir.glob(
            f"{self._FILE_PREFIX_PATTERN}.nc",
        )

        # Save all downloaded variables (there might be more than the one
        # requested; this will potentially save lots of time)
        grouped_files = defaultdict(list)
        formatted_files: list[Path] = []
        for raw_file in raw_files:
            var_name = raw_file.stem.split("_")[-2]
            grouped_files[var_name].append(raw_file)
        for var_name, var_files in grouped_files.items():
            data_dir = self._get_data_dir(data_root_dir, var_name)

            # Do not overwrite existing other variables
            if var_name != variable.var_name and list(data_dir.glob("*.nc")):
                continue
            logger.debug(f"Downloaded and formatted '{var_name}' of {self.id}")
            data_dir.mkdir(parents=True, exist_ok=True)

            for var_file in var_files:
                formatted_file = data_dir / var_file.name
                shutil.move(var_file, formatted_file)
                formatted_files.append(formatted_file)

        return formatted_files

    @contextlib.contextmanager
    def _set_esmvaltool_output_dir(self, output_dir: Path) -> Generator[None]:
        """Set up ESMValTool output directory."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(setup_esmvaltool_config_and_logging())
            stack.enter_context(CFG.context(output_dir=output_dir))
            yield
