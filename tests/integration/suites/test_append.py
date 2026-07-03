from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
from iris.cube import CubeList

from tests._utils import get_all_table_names, get_test_suite

if TYPE_CHECKING:
    from pathlib import Path

    from climateeval.data import DataSourceInformation
    from climateeval.suites import Suite

# The output tables every diagnostic writes (fields of DiagnosticOutput).
_OUTPUT_TABLES = frozenset({"data_sources", "metrics", "raw_output", "variables"})


def _diagnostic_row_counts(
    connection: ibis.BaseBackend,
    suite: Suite,
) -> dict[tuple[str, str], int]:
    """Row count per (diagnostic, table), ignoring system/temporary tables."""
    diagnostics = set(suite._get_diagnostics())
    return {
        (database, table): int(
            connection.table(table, database=database).count().execute(),
        )
        for database, table in get_all_table_names(connection)
        if database in diagnostics and table in _OUTPUT_TABLES
    }


def _copy(data: CubeList) -> CubeList:
    # Each get_database run may modify cubes in place during preprocessing, so
    # hand every run its own copies.
    return CubeList([cube.copy() for cube in data])


def test_get_database_append_accumulates(
    data: CubeList,
    data_information: DataSourceInformation,
    tmp_path: Path,
) -> None:
    """append=True adds to existing schemas instead of failing on them."""
    suite = get_test_suite("simple_diagnostics.yml")
    database_resource = f"duckdb://{tmp_path / 'out.ddb'}"

    suite.get_database(
        _copy(data),
        data_information,
        database_resource=database_resource,
    )
    after_first = _diagnostic_row_counts(ibis.connect(database_resource), suite)
    assert after_first, "first run should have produced at least one table"

    # A second run with append=True must not raise on the existing schemas and
    # must accumulate rows rather than replace them.
    connection = suite.get_database(
        _copy(data),
        data_information,
        database_resource=database_resource,
        append=True,
    )
    after_second = _diagnostic_row_counts(connection, suite)

    assert after_second.keys() == after_first.keys()
    for key, count in after_first.items():
        assert after_second[key] == 2 * count, key


def test_get_database_append_on_empty_database(
    data: CubeList,
    data_information: DataSourceInformation,
    tmp_path: Path,
) -> None:
    """append=True against a fresh database behaves like a normal write."""
    suite = get_test_suite("simple_diagnostics.yml")

    without_append = suite.get_database(
        _copy(data),
        data_information,
        database_resource=f"duckdb://{tmp_path / 'plain.ddb'}",
    )
    with_append = suite.get_database(
        _copy(data),
        data_information,
        database_resource=f"duckdb://{tmp_path / 'fresh.ddb'}",
        append=True,
    )

    assert _diagnostic_row_counts(with_append, suite) == _diagnostic_row_counts(
        without_append,
        suite,
    )
