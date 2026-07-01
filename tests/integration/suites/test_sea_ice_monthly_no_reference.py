from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._utils import get_all_table_names, get_test_suite

if TYPE_CHECKING:
    import ibis
    from iris.cube import CubeList

    from climateeval.data import DataSourceInformation

_DIAGNOSTICS = ["annual_cycle", "annual_mean_timeseries"]


@pytest.fixture(scope="module")
def database(
    data: CubeList,
    data_information: DataSourceInformation,
) -> ibis.BaseBackend:
    suite = get_test_suite("sea_ice_monthly_no_reference.yml")
    return suite.get_database(data, data_information)


@pytest.mark.parametrize("diagnostic", _DIAGNOSTICS)
def test_no_metrics(diagnostic: str, database: ibis.BaseBackend) -> None:
    assert (diagnostic, "metrics") not in get_all_table_names(database)


@pytest.mark.parametrize("table", ["data_sources", "raw_output", "variables"])
@pytest.mark.parametrize("diagnostic", _DIAGNOSTICS)
def test_tables_exist(
    diagnostic: str,
    table: str,
    database: ibis.BaseBackend,
) -> None:
    assert (diagnostic, table) in get_all_table_names(database)
