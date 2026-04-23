from __future__ import annotations

from importlib import resources
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import pytest
from iris.cube import CubeList

from benchmarking.data import DataSourceInformation
from benchmarking.suites import Suite
from tests._utils import assert_table_equal, get_all_table_names
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)

SUITES_YML_DIR: Path = (
    Path(str(resources.files("tests.integration.suites"))) / "suites_yml"
)


@pytest.fixture
def cubes() -> CubeList:
    tas_cube = get_simple_2d_cube(
        var_name="tas",
        standard_name="air_temperature",
        long_name="Near-Surface Air Temperature",
        units="K",
    )
    pr_cube = get_simple_2d_cube(
        var_name="pr",
        standard_name="precipitation_flux",
        long_name="Precipitation",
        units="kg m-2 s-1",
    )
    z_coord = get_air_pressure_coord()
    ta_cube = get_simple_3d_cube(
        z_coord,
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
    )
    return CubeList([tas_cube, pr_cube, ta_cube])


@pytest.fixture
def data_information() -> DataSourceInformation:
    return DataSourceInformation(name="N", category="C", exp="E", variant="V")


def test_simple_diagnostics(
    sample_data_root_dir: Path,
    cubes: CubeList,
    data_information: DataSourceInformation,
) -> None:
    suite = Suite(
        SUITES_YML_DIR / "simple_diagnostics.yml",
        diagnostic_kwargs={
            "data_root_dir": sample_data_root_dir,
            "fail_on_missing_data": True,
        },
    )

    con = suite.get_database(cubes, data_information)
    table_names = get_all_table_names(con)

    expected_diagnostics = ("AnnualCycle", "TimeSeries")
    for diagnostic in expected_diagnostics:
        assert (diagnostic, "data_sources") in table_names
        assert (diagnostic, "metrics") in table_names
        assert (diagnostic, "raw_output") in table_names
        assert (diagnostic, "variables") in table_names

        # Check data source tables (this is identical across diagnostics)
        expected_data_sources = ibis.memtable(
            pd.DataFrame(
                {
                    "id": pd.Series(
                        ["C_N_E_V", "reanalysis_ERA5", "observation_GPCP"],
                        dtype="str",
                    ),
                    "name": pd.Series(["N", "ERA5", "GPCP"], dtype="str"),
                    "category": pd.Series(
                        ["C", "reanalysis", "observation"],
                        dtype="str",
                    ),
                    "institute": pd.Series(
                        ["", "ECMWF", "NASA-GSFC"],
                        dtype="str",
                    ),
                    "exp": pd.Series(["E", "", ""], dtype="str"),
                    "variant": pd.Series(["V", "", ""], dtype="str"),
                    "references": pd.Series(
                        [
                            "",
                            "https://doi.org/10.24381/cds.f17050d7 https://doi.org/10.24381/cds.6860a573",
                            "https://doi.org/10.5067/MEASURES/GPCP/DATA306 https://doi.org/10.5067/MEASURES/GPCP/DATA307",
                        ],
                        dtype="str",
                    ),
                },
            ),
        )
        assert_table_equal(
            con.table("data_sources", database=diagnostic),
            expected_data_sources,
        )

    # Check variables
    expected_variables_annual_cycle = ibis.memtable(
        pd.DataFrame(
            {
                "id": pd.Series(["tas", "mypr"], dtype="str"),
                "var_name": pd.Series(["tas", "pr"], dtype="str"),
                "frequency": pd.Series(["mon", "mon"], dtype="str"),
                "vertical_level": pd.Series(["", ""], dtype="str"),
                "min_lat": pd.Series([-90.0, -10.0], dtype="float64"),
                "max_lat": pd.Series([90.0, 40.0], dtype="float64"),
                "min_lon": pd.Series([0.0, 10.0], dtype="float64"),
                "max_lon": pd.Series([360.0, 200.0], dtype="float64"),
                "landsea_mask": pd.Series(["*", "*"], dtype="str"),
                "ar6_regions_mask": pd.Series(["", ""], dtype="str"),
                "timerange": pd.Series(["*", "19700101/19701231"], dtype="str"),
                "units": pd.Series(["K", "g m-2 s-1"], dtype="str"),
                "anomaly": pd.Series(["", ""], dtype="str"),
                "coordinates": pd.Series(["time lat lon", "time lat lon"], dtype="str"),
                "derived": pd.Series([False, False], dtype="bool"),
                "long_name": pd.Series(
                    ["Near-Surface Air Temperature", "Precipitation"],
                    dtype="str",
                ),
                "realm": pd.Series(["atmos", "atmos"], dtype="str"),
                "standard_name": pd.Series(
                    ["air_temperature", "precipitation_flux"],
                    dtype="str",
                ),
                "cmip6_table_id": pd.Series(["Amon", "Amon"], dtype="str"),
            },
        ),
    )
    assert_table_equal(
        con.table("variables", database="AnnualCycle"),
        expected_variables_annual_cycle,
    )
    expected_variables_timeseries = ibis.memtable(
        pd.DataFrame(
            {
                "id": pd.Series(["ta1000", "mypr"], dtype="str"),
                "var_name": pd.Series(["ta", "pr"], dtype="str"),
                "frequency": pd.Series(["mon", "mon"], dtype="str"),
                "vertical_level": pd.Series(
                    ["air_pressure = 1000.0 hPa", ""],
                    dtype="str",
                ),
                "min_lat": pd.Series([-90.0, -90.0], dtype="float64"),
                "max_lat": pd.Series([90.0, 90.0], dtype="float64"),
                "min_lon": pd.Series([0.0, 0.0], dtype="float64"),
                "max_lon": pd.Series([360.0, 360.0], dtype="float64"),
                "landsea_mask": pd.Series(["*", "*"], dtype="str"),
                "ar6_regions_mask": pd.Series(["", ""], dtype="str"),
                "timerange": pd.Series(
                    ["19700101/19711231", "19710101/19721231"],
                    dtype="str",
                ),
                "units": pd.Series(["K", "g m-2 s-1"], dtype="str"),
                "anomaly": pd.Series(["", "full"], dtype="str"),
                "coordinates": pd.Series(
                    ["time alevel lat lon", "time lat lon"],
                    dtype="str",
                ),
                "derived": pd.Series([False, False], dtype="bool"),
                "long_name": pd.Series(
                    ["Air Temperature", "Precipitation"],
                    dtype="str",
                ),
                "realm": pd.Series(["atmos", "atmos"], dtype="str"),
                "standard_name": pd.Series(
                    ["air_temperature", "precipitation_flux"],
                    dtype="str",
                ),
                "cmip6_table_id": pd.Series(["Amon", "Amon"], dtype="str"),
            },
        ),
    )
    assert_table_equal(
        con.table("variables", database="TimeSeries"),
        expected_variables_timeseries,
    )

    # Check raw output
    expected_raw_output_annual_cycle = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    [
                        "C_N_E_V",
                        "C_N_E_V",
                        "reanalysis_ERA5",
                        "reanalysis_ERA5",
                        "observation_GPCP",
                        "reanalysis_ERA5",
                    ],
                    dtype="str",
                ),
                "data_type": pd.Series(
                    [
                        "to_benchmark",
                        "to_benchmark",
                        "reference",
                        "reference",
                        "reference",
                        "other",
                    ],
                    dtype="str",
                ),
                "month_number": pd.Series(
                    [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
                    dtype="float32",
                ),
                "mypr": pd.Series(
                    [3476.613525, np.nan, np.nan, np.nan, 6953.227051, 3476.613525],
                    dtype="float32",
                ),
                "tas": pd.Series(
                    [5.5, 14.5, 11.0, 29.0, np.nan, np.nan],
                    dtype="float32",
                ),
            },
        ),
    )
    assert_table_equal(
        con.table("raw_output", database="AnnualCycle"),
        expected_raw_output_annual_cycle,
    )
    expected_raw_output_timeseries = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    [
                        "C_N_E_V",
                        "C_N_E_V",
                        "C_N_E_V",
                        "reanalysis_ERA5",
                        "reanalysis_ERA5",
                        "observation_GPCP",
                        "observation_GPCP",
                        "reanalysis_ERA5",
                        "reanalysis_ERA5",
                    ],
                    dtype="str",
                ),
                "data_type": pd.Series(
                    [
                        "to_benchmark",
                        "to_benchmark",
                        "to_benchmark",
                        "reference",
                        "reference",
                        "reference",
                        "reference",
                        "other",
                        "other",
                    ],
                    dtype="str",
                ),
                "time": pd.Series(
                    [
                        "1970-07-01 00:00:00",
                        "1971-07-01 00:00:00",
                        "1972-07-01 00:00:00",
                        "1970-07-01 00:00:00",
                        "1971-07-01 00:00:00",
                        "1971-07-01 00:00:00",
                        "1972-07-01 00:00:00",
                        "1971-07-01 00:00:00",
                        "1972-07-01 00:00:00",
                    ],
                    dtype="str",
                ),
                "mypr": pd.Series(
                    [
                        np.nan,
                        -2900.0,
                        3100.0,
                        np.nan,
                        np.nan,
                        -5800.0,
                        6200.0,
                        -2900.0,
                        3100.0,
                    ],
                    dtype="float32",
                ),
                "ta1000": pd.Series(
                    [2.5, 14.5, np.nan, 5.0, 29.0, np.nan, np.nan, np.nan, np.nan],
                    dtype="float32",
                ),
            },
        ),
    )
    expected_raw_output_timeseries = expected_raw_output_timeseries.cast(
        time="timestamp(6)",
    )
    assert_table_equal(
        con.table("raw_output", database="TimeSeries"),
        expected_raw_output_timeseries,
    )

    # Check metrics
    expected_metrics_annual_cycle = ibis.memtable(
        pd.DataFrame(
            {
                "var_id": pd.Series(["tas", "mypr", "mypr"], dtype="str"),
                "weighted_rmse": pd.Series(
                    [10.965857, 3476.613525, 3476.613525],
                    dtype="float32",
                ),
                "weighted_pearsonr": pd.Series([1.0, np.nan, np.nan], dtype="float32"),
                "weighted_emd": pd.Series(
                    [9.87, 3441.8474, 3441.8474],
                    dtype="float32",
                ),
                "data_id": pd.Series(
                    ["C_N_E_V", "C_N_E_V", "reanalysis_ERA5"],
                    dtype="str",
                ),
                "reference_data_id": pd.Series(
                    ["reanalysis_ERA5", "observation_GPCP", "observation_GPCP"],
                    dtype="str",
                ),
            },
        ),
    )
    assert_table_equal(
        con.table("metrics", database="AnnualCycle"),
        expected_metrics_annual_cycle,
    )
    expected_metrics_timeseries = ibis.memtable(
        pd.DataFrame(
            {
                "var_id": pd.Series(["ta1000", "mypr", "mypr"], dtype="str"),
                "weighted_rmse": pd.Series(
                    [10.404326, 3001.802979, 3001.802979],
                    dtype="float32",
                ),
                "weighted_pearsonr": pd.Series([1.0, 1.0, 1.0], dtype="float32"),
                "weighted_emd": pd.Series(
                    [8.3475, 2940.082, 2940.082],
                    dtype="float32",
                ),
                "data_id": pd.Series(
                    ["C_N_E_V", "C_N_E_V", "reanalysis_ERA5"],
                    dtype="str",
                ),
                "reference_data_id": pd.Series(
                    ["reanalysis_ERA5", "observation_GPCP", "observation_GPCP"],
                    dtype="str",
                ),
            },
        ),
    )
    assert_table_equal(
        con.table("metrics", database="TimeSeries"),
        expected_metrics_timeseries,
    )
