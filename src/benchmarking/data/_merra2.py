"""Provide MERRA2 data."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from esmvaltool.cmorizers.data.downloaders.wget import NASADownloader

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from benchmarking import Variable


class MERRA2(DataSource):
    """Manage MERRA2 data."""

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data.

        Note
        ----
        This requires valid credentials in ``~/.netrc`` of the form:

        ```netrc
        machine urs.earthdata.nasa.gov login <LOGIN> password <PASSWORD>
        ```

        Make sure that ~/.netrc is only readable by yourself, i.e., run

        ```bash
        chmod 600 ~/.netrc`
        ```

        An account can be registered at https://urs.earthdata.nasa.gov/.

        """
        supported_var_names = [
            "evspsbl",
            "hfls",
            "hfss",
            "pr",
            "prc",
            "prsn",
            "tauu",
            "tauv",
        ]
        self._check_supported_variables(variable, supported_var_names)

        # Raw downloaded data by ESMValTool will be stored in temporary
        # directory. Our "raw" data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Manually download required files because regular ESMValTool
        # downloader downloads all variables at once (takes lots of time)
        tier = 3
        source = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/"
        dataset_info = {
            "tier": tier,
            "source": source,
            "last_access": datetime.now(UTC),
            "info": "",
        }
        downloader = NASADownloader(
            original_data_dir=tmp_dir,
            dataset=self.name,
            dataset_info=dataset_info,
            overwrite=False,
        )

        for year in range(1980, 2022):
            downloader.download_folder(f"{source}/M2TMNXFLX.5.12.4/{year}/")

        return self._esmvaltool_cmorizer_format(
            data_root_dir,
            variable,
            download_dir=tmp_dir,
            tier=tier,
        )
