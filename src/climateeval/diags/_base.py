"""Provide diagnostic base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import ibis
import iris
import pandas as pd
from ibis import _
from iris.pandas import as_data_frame
from loguru import logger

from climateeval import NO_LEVEL
from climateeval.exceptions import MissingDataError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.data import DataSourceInformation

if TYPE_CHECKING:
    from iris.cube import CubeList
    from xarray import Dataset

    from climateeval.data import DataSourceInformation

logger = logger.opt(colors=True)


_DEFAULT_DATA_ROOT_DIR: Path = Path.home() / ".climateeval" / "data"


class Diagnostic(ABC):
    """Manage diagnostics."""

    _DATA_ID_COLUMN: ClassVar[str] = "data_id"
    """Name of table column containing data ID."""

    _REFERENCE_DATA_ID_COLUMN: ClassVar[str] = "reference_data_id"
    """Name of table column containing reference data ID."""

    _DATA_TYPE_COLUMN: ClassVar[str] = "data_type"
    """Name of table column containing data type."""

    _DTYPE: ClassVar[ibis.DataType] = "float32"
    """Dtype of numerical values."""

    _VARIABLE_ID_COLUMN: ClassVar[str] = "var_id"
    """Name of table column containing variable ID."""

    # Attributes that subclass instances need to define

    _data_root_dir: Path
    """Data root directory."""

    _download_missing_data: bool
    """Download missing data."""

    _fail_on_missing_data: bool
    """Fail on missing data."""

    @property
    def data_root_dir(self) -> Path:
        """Get path to data root directory."""
        return self._data_root_dir

    @property
    def name(self) -> str:
        """Get diagnostic name."""
        return self.__class__.__name__

    @classmethod
    @abstractmethod
    def from_suite_dict(
        cls,
        suite_settings: dict[str, Any] | None,
        additional_variable_kwargs: dict[str, Any],
        **additional_diagnostic_kwargs: Any,
    ) -> Self:
        """Load diagnostic from suite dictionary (given by YAML file)."""

    @abstractmethod
    def get_output(
        self,
        data: dict[str, CubeList | Dataset],
        data_information: DataSourceInformation,
    ) -> DiagnosticOutput:
        """Get diagnostic output."""

    def _get_data_root_dir(self, data_root_dir: Path | None) -> Path:
        """Get data root directory."""
        if data_root_dir is None:
            data_root_dir = _DEFAULT_DATA_ROOT_DIR
        data_root_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using data root directory {data_root_dir}")
        return data_root_dir

    def _get_data_sources_table(
        self,
        data_source_information: Iterable[DataSourceInformation],
    ) -> ibis.Table:
        """Get data sources table."""
        df = pd.DataFrame(sorted(set(data_source_information)))
        df["references"] = df["references"].map(" ".join)
        return ibis.memtable(df)

    def _get_raw_output_table(
        self,
        cube: Cube,
        *,
        variable: Variable,
        data_id: str,
        data_type: Literal["to_benchmark", "reference", "other"],
    ) -> ibis.Table:
        """Get data table of a single cube."""
        with iris.FUTURE.context(pandas_ndim=True):
            df = as_data_frame(cube).reset_index()
        if "time" in df.columns:
            df["time"] = df["time"].astype(str)
        raw_output_table = ibis.memtable(df)
        cast_operations = {
            c: {"time": "timestamp(6)"}.get(c, self._DTYPE)
            for c in raw_output_table.columns
        }
        raw_output_table = raw_output_table.cast(**cast_operations)
        raw_output_table = raw_output_table.rename(**{variable.id: cube.name()})
        raw_output_table = raw_output_table.mutate(
            **{self._DATA_ID_COLUMN: ibis.literal(data_id)},
        )
        raw_output_table = raw_output_table.mutate(
            **{self._DATA_TYPE_COLUMN: ibis.literal(data_type)},
        )
        return raw_output_table  # noqa: RET504

    def _get_variables_table(self, variables: Iterable[Variable]) -> ibis.Table:
        """Get variables table."""
        df = pd.DataFrame(sorted(set(variables)))
        df = df.drop("settings", axis=1)
        df["coordinates"] = df["coordinates"].map(
            lambda r: " ".join(c["var_name"] for c in r),
        )
        df["ar6_regions_mask"] = df["ar6_regions_mask"].map(" ".join)

        def vertical_level_dict_to_str(
            vertical_level_dict: dict[str, str | float],
        ) -> str:
            """Convert vertical level dictionary to string."""
            if vertical_level_dict == asdict(NO_LEVEL):
                return ""
            return (
                f"{vertical_level_dict['coordinate_name']} = "
                f"{vertical_level_dict['value']} {vertical_level_dict['units']}"
            )

        df["vertical_level"] = df["vertical_level"].map(vertical_level_dict_to_str)
        return ibis.memtable(df)

    def _handle_missing_data(
        self,
        data_id: str,
        variables: Iterable[Variable],
        exception: Exception,
    ) -> None:
        """Check if data is available."""
        msg = f"Failed to get data {data_id} for variable(s) {variables}: {exception}"
        if self._fail_on_missing_data:
            raise MissingDataError(msg) from exception
        logger.warning(msg)

    def _remove_superfluous_rows(
        self,
        table: ibis.Table,
        *,
        unique_columns: Iterable[str],
        variable_columns: Iterable[str],
    ) -> ibis.Table:
        """Drop superfluous rows by removing NaN entries.

        Note
        ----
        Currently, we first add NaNs and then remove it again. The reason for
        this is that the logic is simple and the code robust. There are
        certainly smarter and more efficient ways to do this, but since our
        tables are usually quite small, it's probably not necessary.

        """
        variable_columns = tuple(variable_columns)
        table = table.group_by(unique_columns).aggregate(
            **{v: _[v].first() for v in variable_columns},
        )
        return table.drop_null(variable_columns, how="all")


@dataclass(frozen=True, kw_only=True)
class DiagnosticOutput:
    """Manage diagnostic output."""

    raw_output: ibis.Table | None
    metrics: ibis.Table | None
    variables: ibis.Table
    data_sources: ibis.Table
