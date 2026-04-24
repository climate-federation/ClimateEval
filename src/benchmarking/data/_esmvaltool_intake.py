"""Provide ESMValTool Intake-ESGF data sources."""

from __future__ import annotations

import contextlib
import tempfile
from collections import defaultdict
from copy import deepcopy
from importlib import resources
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, ClassVar

import intake_esgf
import iris
import yaml
from esmvalcore.config import CFG
from esmvalcore.dataset import Dataset
from loguru import logger

from benchmarking import setup_esmvaltool_config_and_logging
from benchmarking.data._base import DataSource, DataSourceGenerator
from benchmarking.data._utils import DataSourceInformation, get_cmip6_reference

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from benchmarking import Variable


logger = logger.opt(colors=True)


ESMVALTOOL_INTAKE_DATA_SOURCES = (
    Path(str(resources.files("benchmarking.data")))
    / "config"
    / "esmvaltool_intake_data_sources.yml"
)


def _get_esmvaltool_intake_data_sources() -> dict[str, Any]:
    """Get ESMValTool intake-esgf data sources."""
    with ESMVALTOOL_INTAKE_DATA_SOURCES.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@contextlib.contextmanager
def _setup_esmvaltool_intake_data_source(cache_dir: Path) -> Generator[None]:
    """Set up ESMValTool intake-esgf data sources."""
    intake_esgf_options = {
        "all_indices": True,
        "esg_dataroot": [],
        "local_cache": cache_dir,
    }

    with contextlib.ExitStack() as stack:
        stack.enter_context(intake_esgf.conf.set(**intake_esgf_options))
        stack.enter_context(setup_esmvaltool_config_and_logging())
        esmvaltool_config = deepcopy(CFG)
        for (
            project,
            data_config,
        ) in _get_esmvaltool_intake_data_sources().items():
            esmvaltool_config["projects"][project]["data"] = data_config["data"]
        stack.enter_context(CFG.context(esmvaltool_config))
        yield


class ESMValToolIntakeDataSource(DataSource):
    """Manage ESMValTool Intake-ESGF data source."""

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {}
    """Facets used to download data."""

    def download(self, variable: Variable, data_root_dir: Path) -> list[Path]:
        """Download data.

        See
        https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html#data-sources.

        """
        facets = {
            "dataset": self.name,
            "mip": variable.cmip6_table_id,
            "short_name": variable.var_name,
            "frequency": variable.frequency,
        }
        facets.update(self.ESMVALTOOL_DATASET_FACETS)

        # Store downloaded data in temporary directory
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with _setup_esmvaltool_intake_data_source(tmp_dir):
            cube = Dataset(**facets).load()

        # Save file into data directory
        data_dir = self._get_data_dir(data_root_dir, variable.var_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / self._get_output_filename(cube, **facets)
        iris.save(cube, output_path)

        return [output_path]


class ESMValToolIntakeDataSourceGenerator(DataSourceGenerator):
    """Generate ESMValTool Intake-ESGF data sources."""

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {}
    """Facets used to generate data sources."""

    def generate(
        self,
        variable: Variable,
    ) -> Generator[ESMValToolIntakeDataSource]:
        """Generate data sources."""
        facets = {
            "mip": variable.cmip6_table_id,
            "short_name": variable.var_name,
            "frequency": variable.frequency,
        }
        facets.update(self.ESMVALTOOL_DATASET_FACETS)

        logger.debug(
            f"Generating data sources for {variable} using ESMValTool "
            f"Intake-ESGF facets\n{pformat(facets)}",
        )

        for idx, dataset in enumerate(Dataset(**facets).from_files()):
            new_esmvaltool_dataset_facets = dict(dataset.facets)
            reference = get_cmip6_reference(
                mip="CMIP",
                institute=new_esmvaltool_dataset_facets["institute"],
                model=new_esmvaltool_dataset_facets["dataset"],
            )
            new_data_source_information = DataSourceInformation(
                name=new_esmvaltool_dataset_facets["dataset"],
                category=new_esmvaltool_dataset_facets["project"],
                institute=new_esmvaltool_dataset_facets["institute"],
                exp=new_esmvaltool_dataset_facets["exp"],
                variant=new_esmvaltool_dataset_facets["ensemble"],
                references=(reference,),
            )
            class_name = f"{self.__class__.__name__}DataSource{idx}"

            # Dynamically create subclass for each data set that has been found
            new_data_source_cls = type(
                class_name,
                (ESMValToolIntakeDataSource,),
                {
                    "_INFORMATION": new_data_source_information,
                    "ESMVALTOOL_DATASET_FACETS": new_esmvaltool_dataset_facets,
                },
            )
            yield new_data_source_cls()


def filter_intake_data_sources(
    *generators: Generator[ESMValToolIntakeDataSource],
    group_by: str,
    required_facets: Iterable[dict[str, str]],
) -> dict[str, list[ESMValToolIntakeDataSource]]:
    """Filter out data sources where not all required facets are available.

    For each group, at least one data source per required facet must be
    present. Otherwise, this group is filtered out. Data sources are identified
    via their :attr:`~ESMValToolIntakeDataSource.ESMVALTOOL_DATASET_FACETS`
    attribute.

    Parameters
    ----------
    *generators:
        Generators that produce data sources. Can be the return value of
        :meth:`ESMValToolIntakeDataSource.generate`.
    group_by:
        Facets used to identify group elements. Values of these facets will be
        the dictionary keys of the return value of this function.
    required_facets:
        Required facet combinations. To not be filtered out, at least one data
        source per facet combination must be present per group.

    Returns
    -------
    :
        Filtered data sources.

    Example
    -------

    ```py
    group_by = "dataset"
    required_facets = [
        {"exp": "piControl", "short_name": "tas"},
        {"exp": "historical", "short_name": "pr"},
    ]
    facets_of_input_data_sources = [  # these are produced by the generators
        {"dataset": "A", "exp": "piControl", "short_name": "tas"},
        {"dataset": "A", "exp": "historical", "short_name": "pr"},
        {"dataset": "B", "exp": "historical", "short_name": "pr"},
        {"dataset": "C", "exp": "piControl", "short_name": "pr"},
    ]

    filtered_output = filter_intake_data_sources(
        *generators, group_by=group_by, required_facets=required_facets
    )

    # Facets of filtered_output
    {
        "A": [
            {"dataset": "A", "exp": "piControl", "short_name": "tas"},
            {"dataset": "A", "exp": "historical", "short_name": "pr"},
        ],
    }
    ```

    """
    # Generate all data sources and group them by group_by facet
    all_data_sources: dict[str, list[ESMValToolIntakeDataSource]] = defaultdict(list)
    for generator in generators:
        for data_source in generator:
            if group_by in data_source.ESMVALTOOL_DATASET_FACETS:
                all_data_sources[
                    data_source.ESMVALTOOL_DATASET_FACETS[group_by]
                ].append(
                    data_source,
                )

    # Check if each group contains all required facets
    filtered_data_sources: dict[str, list[ESMValToolIntakeDataSource]] = {}
    for group, data_sources in all_data_sources.items():
        for required_facet_dict in required_facets:
            data_sources_facets = tuple(
                {(f, d.ESMVALTOOL_DATASET_FACETS.get(f)) for f in required_facet_dict}
                for d in data_sources
            )
            required_facet_set = set(required_facet_dict.items())
            if required_facet_set not in data_sources_facets:
                break
        else:
            filtered_data_sources[group] = data_sources

    return filtered_data_sources
