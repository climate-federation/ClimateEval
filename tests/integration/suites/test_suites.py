from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import ibis
import numpy as np
import pandas as pd

from climateeval.suites import Suite
from tests._utils import assert_table_equal, get_all_table_names

if TYPE_CHECKING:
    from iris.cube import CubeList

    from climateeval.data import DataSourceInformation

SUITES_YML_DIR: Path = (
    Path(str(resources.files("tests.integration.suites"))) / "suites_yml"
)


def test_simple_diagnostics(
    sample_data_root_dir: Path,
    data: CubeList,
    data_information: DataSourceInformation,
) -> None:
    suite = Suite(
        SUITES_YML_DIR / "simple_diagnostics.yml",
        diagnostic_kwargs={
            "data_root_dir": sample_data_root_dir,
            "fail_on_missing_data": True,
            "download_missing_data": False,
        },
    )

    con = suite.get_database(data, data_information)
    table_names = get_all_table_names(con)

    expected_diagnostics = ("annual_cycle", "annual_mean_timeseries")
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
                            "https://doi.org/10.24381/cds.f17050d7 "
                            "https://doi.org/10.24381/cds.6860a573 "
                            "https://doi.org/10.24381/cds.adbb2d47",
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
                "diagnostic_kwargs": pd.Series(["", ""], dtype="str"),
                "additional_preprocessors": pd.Series(["", ""], dtype="str"),
                "reasonable_min": pd.Series([np.nan, np.nan], dtype="float64"),
                "reasonable_max": pd.Series([np.nan, np.nan], dtype="float64"),
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
                "ndim": pd.Series([3, 3], dtype="int"),
            },
        ),
    )
    assert_table_equal(
        con.table("variables", database="annual_cycle"),
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
                "diagnostic_kwargs": pd.Series(["", ""], dtype="str"),
                "additional_preprocessors": pd.Series(["", ""], dtype="str"),
                "reasonable_min": pd.Series([np.nan, np.nan], dtype="float64"),
                "reasonable_max": pd.Series([np.nan, np.nan], dtype="float64"),
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
                "ndim": pd.Series([4, 3], dtype="int"),
            },
        ),
    )
    assert_table_equal(
        con.table("variables", database="annual_mean_timeseries"),
        expected_variables_timeseries,
    )

    # Check raw output
    expected_raw_output_annual_cycle = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    ["C_N_E_V"] * 2
                    + ["reanalysis_ERA5"] * 2
                    + [
                        "observation_GPCP",
                        "reanalysis_ERA5",
                    ],
                    dtype="str",
                ),
                "data_type": pd.Series(
                    ["to_benchmark"] * 2
                    + ["reference"] * 2
                    + [
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
        con.table("raw_output", database="annual_cycle"),
        expected_raw_output_annual_cycle,
    )
    expected_raw_output_timeseries = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    ["C_N_E_V"] * 3
                    + ["reanalysis_ERA5"] * 2
                    + ["observation_GPCP"] * 2
                    + ["reanalysis_ERA5"] * 2,
                    dtype="str",
                ),
                "data_type": pd.Series(
                    ["to_benchmark"] * 3 + ["reference"] * 4 + ["other"] * 2,
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
                        8500.0,
                        14500.0,
                        np.nan,
                        np.nan,
                        17000.0,
                        29000.0,
                        8500.0,
                        14500.0,
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
        con.table("raw_output", database="annual_mean_timeseries"),
        expected_raw_output_timeseries,
    )

    # Check metrics
    expected_metrics_annual_cycle = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    ["C_N_E_V", "C_N_E_V", "reanalysis_ERA5"],
                    dtype="str",
                ),
                "reference_data_id": pd.Series(
                    ["reanalysis_ERA5", "observation_GPCP", "observation_GPCP"],
                    dtype="str",
                ),
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
            },
        ),
    )
    assert_table_equal(
        con.table("metrics", database="annual_cycle"),
        expected_metrics_annual_cycle,
    )
    expected_metrics_timeseries = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    ["C_N_E_V", "C_N_E_V", "reanalysis_ERA5"],
                    dtype="str",
                ),
                "reference_data_id": pd.Series(
                    ["reanalysis_ERA5", "observation_GPCP", "observation_GPCP"],
                    dtype="str",
                ),
                "var_id": pd.Series(["ta1000", "mypr", "mypr"], dtype="str"),
                "weighted_rmse": pd.Series(
                    [10.404326, 11888.835, 11888.835],
                    dtype="float32",
                ),
                "weighted_pearsonr": pd.Series([1.0, 1.0, 1.0], dtype="float32"),
                "weighted_emd": pd.Series(
                    [8.3475, 11381.566, 11381.566],
                    dtype="float32",
                ),
            },
        ),
    )
    assert_table_equal(
        con.table("metrics", database="annual_mean_timeseries"),
        expected_metrics_timeseries,
    )


def test_simple_diagnostics_no_reference(
    sample_data_root_dir: Path,
    data: CubeList,
    data_information: DataSourceInformation,
) -> None:
    suite = Suite(
        SUITES_YML_DIR / "simple_diagnostics_no_reference.yml",
        diagnostic_kwargs={
            "data_root_dir": sample_data_root_dir,
            "fail_on_missing_data": True,
            "download_missing_data": False,
        },
    )

    con = suite.get_database(data, data_information)
    table_names = get_all_table_names(con)

    assert ("annual_cycle", "data_sources") in table_names
    assert ("annual_cycle", "metrics") not in table_names
    assert ("annual_cycle", "raw_output") in table_names
    assert ("annual_cycle", "variables") in table_names

    # Check data sources
    expected_data_sources = ibis.memtable(
        pd.DataFrame(
            {
                "id": pd.Series(["C_N_E_V"], dtype="str"),
                "name": pd.Series(["N"], dtype="str"),
                "category": pd.Series(["C"], dtype="str"),
                "institute": pd.Series([""], dtype="str"),
                "exp": pd.Series(["E"], dtype="str"),
                "variant": pd.Series(["V"], dtype="str"),
                "references": pd.Series([""], dtype="str"),
            },
        ),
    )
    assert_table_equal(
        con.table("data_sources", database="annual_cycle"),
        expected_data_sources,
    )

    # Check variables
    expected_variables = ibis.memtable(
        pd.DataFrame(
            {
                "id": pd.Series(["tas", "asr"], dtype="str"),
                "var_name": pd.Series(["tas", "asr"], dtype="str"),
                "frequency": pd.Series(["mon", "mon"], dtype="str"),
                "vertical_level": pd.Series(["", ""], dtype="str"),
                "min_lat": pd.Series([-90.0, -90.0], dtype="float64"),
                "max_lat": pd.Series([90.0, 90.0], dtype="float64"),
                "min_lon": pd.Series([0.0, 0.0], dtype="float64"),
                "max_lon": pd.Series([360.0, 360.0], dtype="float64"),
                "landsea_mask": pd.Series(["*", "*"], dtype="str"),
                "ar6_regions_mask": pd.Series(["", ""], dtype="str"),
                "timerange": pd.Series(["*", "*"], dtype="str"),
                "units": pd.Series(["K", "W m-2"], dtype="str"),
                "diagnostic_kwargs": pd.Series(["a=1 b=2", ""], dtype="str"),
                "additional_preprocessors": pd.Series(["", ""], dtype="str"),
                "reasonable_min": pd.Series([np.nan, np.nan], dtype="float64"),
                "reasonable_max": pd.Series([np.nan, np.nan], dtype="float64"),
                "coordinates": pd.Series(["time lat lon", "time lat lon"], dtype="str"),
                "derived": pd.Series([False, True], dtype="bool"),
                "long_name": pd.Series(
                    ["Near-Surface Air Temperature", "Absorbed shortwave radiation"],
                    dtype="str",
                ),
                "realm": pd.Series(["atmos", "atmos"], dtype="str"),
                "standard_name": pd.Series(["air_temperature", ""], dtype="str"),
                "cmip6_table_id": pd.Series(["Amon", "Amon"], dtype="str"),
                "ndim": pd.Series([3, 3], dtype="int"),
            },
        ),
    )
    assert_table_equal(
        con.table("variables", database="annual_cycle"),
        expected_variables,
    )

    # Check raw output
    expected_raw_output = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(["C_N_E_V"] * 2, dtype="str"),
                "data_type": pd.Series(["to_benchmark"] * 2, dtype="str"),
                "month_number": pd.Series([1.0, 2.0], dtype="float32"),
                "asr": pd.Series([0.0, 0.0], dtype="float32"),
                "tas": pd.Series([5.5, 14.5], dtype="float32"),
            },
        ),
    )
    assert_table_equal(
        con.table("raw_output", database="annual_cycle"),
        expected_raw_output,
    )


def test_histogram(
    sample_data_root_dir: Path,
    data: CubeList,
    data_information: DataSourceInformation,
) -> None:
    suite = Suite(
        SUITES_YML_DIR / "histogram.yml",
        diagnostic_kwargs={
            "data_root_dir": sample_data_root_dir,
            "fail_on_missing_data": True,
            "download_missing_data": False,
        },
    )

    con = suite.get_database(data, data_information)
    table_names = get_all_table_names(con)

    assert ("histogram", "data_sources") in table_names
    assert ("histogram", "metrics") not in table_names
    assert ("histogram", "raw_output") in table_names
    assert ("histogram", "variables") in table_names

    # Check data sources
    expected_data_sources = ibis.memtable(
        pd.DataFrame(
            {
                "id": pd.Series(["C_N_E_V", "reanalysis_ERA5"], dtype="str"),
                "name": pd.Series(["N", "ERA5"], dtype="str"),
                "category": pd.Series(["C", "reanalysis"], dtype="str"),
                "institute": pd.Series(["", "ECMWF"], dtype="str"),
                "exp": pd.Series(["E", ""], dtype="str"),
                "variant": pd.Series(["V", ""], dtype="str"),
                "references": pd.Series(
                    [
                        "",
                        "https://doi.org/10.24381/cds.f17050d7 "
                        "https://doi.org/10.24381/cds.6860a573 "
                        "https://doi.org/10.24381/cds.adbb2d47",
                    ],
                    dtype="str",
                ),
            },
        ),
    )
    assert_table_equal(
        con.table("data_sources", database="histogram"),
        expected_data_sources,
    )

    # Check variables
    expected_variables = ibis.memtable(
        pd.DataFrame(
            {
                "id": pd.Series(["tas"], dtype="str"),
                "var_name": pd.Series(["tas"], dtype="str"),
                "frequency": pd.Series(["mon"], dtype="str"),
                "vertical_level": pd.Series([""], dtype="str"),
                "min_lat": pd.Series([-90.0], dtype="float64"),
                "max_lat": pd.Series([90.0], dtype="float64"),
                "min_lon": pd.Series([0.0], dtype="float64"),
                "max_lon": pd.Series([360.0], dtype="float64"),
                "landsea_mask": pd.Series(["*"], dtype="str"),
                "ar6_regions_mask": pd.Series([""], dtype="str"),
                "timerange": pd.Series(["*"], dtype="str"),
                "units": pd.Series(["K"], dtype="str"),
                "diagnostic_kwargs": pd.Series(
                    ["bins=3 bin_range=[0.0, 30.0]"],
                    dtype="str",
                ),
                "additional_preprocessors": pd.Series(
                    [
                        "esmvalcore.preprocessor.extract_region: start_longitude=25.0 "
                        "end_longitude=75.0 start_latitude=-15.0 end_latitude=15.0",
                    ],
                    dtype="str",
                ),
                "reasonable_min": pd.Series([np.nan], dtype="float64"),
                "reasonable_max": pd.Series([np.nan], dtype="float64"),
                "coordinates": pd.Series(["time lat lon"], dtype="str"),
                "derived": pd.Series([False], dtype="bool"),
                "long_name": pd.Series(["Near-Surface Air Temperature"], dtype="str"),
                "realm": pd.Series(["atmos"], dtype="str"),
                "standard_name": pd.Series(["air_temperature"], dtype="str"),
                "cmip6_table_id": pd.Series(["Amon"], dtype="str"),
                "ndim": pd.Series([3], dtype="int"),
            },
        ),
    )
    assert_table_equal(
        con.table("variables", database="histogram"),
        expected_variables,
    )

    # Check raw output
    expected_raw_output = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(
                    ["C_N_E_V"] * 3 + ["reanalysis_ERA5"] * 3,
                    dtype="str",
                ),
                "data_type": pd.Series(
                    ["to_benchmark"] * 3 + ["reference"] * 3,
                    dtype="str",
                ),
                "tas": pd.Series([5.0, 15.0, 25.0] * 2, dtype="float32"),
                "density_tas": pd.Series(
                    [2.0 / 30.0, 1.0 / 30.0, 0.0, 1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0],
                    dtype="float32",
                ),
            },
        ),
    )
    assert_table_equal(
        con.table("raw_output", database="histogram"),
        expected_raw_output,
    )
