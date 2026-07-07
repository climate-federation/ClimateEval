from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._utils import check_table, get_all_table_names, get_test_suite

if TYPE_CHECKING:
    import ibis
    from iris.cube import CubeList
    from pytest_regressions import DataFrameRegressionFixture

    from climateeval.data import DataSourceInformation


@pytest.fixture(scope="module")
def database(
    data: CubeList,
    data_information: DataSourceInformation,
) -> ibis.BaseBackend:
    suite = get_test_suite("simple_diagnostics_no_reference.yml")
    return suite.get_database(data, data_information)


def test_no_metrics(database: ibis.BaseBackend) -> None:
    assert ("annual_cycle", "metrics") not in get_all_table_names(database)


@pytest.mark.parametrize("table", ["data_sources", "raw_output", "variables"])
@pytest.mark.parametrize("diagnostic", ["annual_cycle"])
def test_tables(
    diagnostic: str,
    table: str,
    database: ibis.BaseBackend,
    dataframe_regression: DataFrameRegressionFixture,
) -> None:
    assert (diagnostic, table) in get_all_table_names(database)
    check_table(database.table(table, database=diagnostic), dataframe_regression)
