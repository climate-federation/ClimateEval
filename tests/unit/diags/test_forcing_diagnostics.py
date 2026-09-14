"""Tests for the effective radiative forcing complex diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import DataSourceInformation
from climateeval.diags.complex import AerosolERF, Amip4xCO2ERF, GFMIPPatchFeedback
from climateeval.diags.complex._branch_time import (
    branch_year_in_parent,
    parallel_control_window,
)
from climateeval.diags.complex._forcing import clip_window_to_record

N_LAT = 18
N_LON = 36
DAYS_PER_MONTH = 30.0
DAYS_PER_YEAR = 360.0
CALENDAR = "360_day"


def _info(exp: str = "piControl") -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthModel",
        category="model",
        institute="Synth",
        exp=exp,
        variant="r1i1p1f1",
    )


def _monthly_cube(
    var_name: str,
    units: str,
    values: np.ndarray,
    *,
    first_year: int,
    attributes: dict[str, object] | None = None,
) -> Cube:
    """Spatially uniform monthly cube following an annual series."""
    monthly = np.repeat(np.asarray(values, dtype=float), 12)
    n_time = monthly.size
    time = DimCoord(
        np.arange(n_time, dtype=float) * DAYS_PER_MONTH + 15.0,
        standard_name="time",
        units=Unit(f"days since {first_year}-01-01", calendar=CALENDAR),
    )
    lat = DimCoord(
        np.linspace(-85.0, 85.0, N_LAT),
        standard_name="latitude",
        units="degrees",
    )
    lon = DimCoord(
        np.linspace(5.0, 355.0, N_LON),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    for coord in (time, lat, lon):
        coord.guess_bounds()
    data = np.broadcast_to(
        monthly[:, None, None],
        (n_time, N_LAT, N_LON),
    ).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
        attributes=attributes,
    )


def _experiment(
    toa_net: np.ndarray,
    temperature: np.ndarray | None = None,
    *,
    first_year: int = 1850,
    temperature_name: str = "tas",
    attributes: dict[str, object] | None = None,
) -> CubeList:
    """Cubes realising a TOA net flux series and, optionally, a temperature."""
    toa_net = np.asarray(toa_net, dtype=float)
    cubes = CubeList(
        [
            _monthly_cube(
                "rsdt",
                "W m-2",
                340.0 + toa_net,
                first_year=first_year,
                attributes=attributes,
            ),
            _monthly_cube(
                "rsut",
                "W m-2",
                np.full(toa_net.size, 240.0),
                first_year=first_year,
                attributes=attributes,
            ),
            _monthly_cube(
                "rlut",
                "W m-2",
                np.full(toa_net.size, 100.0),
                first_year=first_year,
                attributes=attributes,
            ),
        ],
    )
    if temperature is not None:
        cubes.append(
            _monthly_cube(
                temperature_name,
                "K",
                temperature,
                first_year=first_year,
                attributes=attributes,
            ),
        )
    return cubes


def _row(output) -> dict:  # noqa: ANN001
    assert output.raw_output is not None
    return output.raw_output.to_pandas().iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Calendar windows and the branch time
# ---------------------------------------------------------------------------


def test_clip_window_to_record() -> None:
    window = (2010, 2019)
    # A record covering the window keeps it
    assert clip_window_to_record(window, 1850, 2020) == (2010, 2019)
    assert clip_window_to_record(window, 1850, 2019) == (2010, 2019)
    # A run ending in 2014 gets the last ten years it has
    assert clip_window_to_record(window, 1850, 2014) == (2005, 2014)
    # The window never slides before the start of the record
    assert clip_window_to_record(window, 2001, 2014) == (2005, 2014)
    # A record shorter than the window is used whole
    assert clip_window_to_record(window, 2012, 2014) == (2012, 2014)
    assert clip_window_to_record(window, 2008, 2014) == (2008, 2014)
    with pytest.raises(ValueError, match="empty window"):
        clip_window_to_record((2019, 2010), 1850, 2020)
    with pytest.raises(ValueError, match="empty record"):
        clip_window_to_record(window, 2020, 1850)


def test_parallel_control_window() -> None:
    # A child starting in 1850 that branched from control year 3200
    assert parallel_control_window(
        (2010, 2019),
        branch_year=3200,
        child_first_year=1850,
    ) == (3360, 3369)
    # A control sharing the child's calendar is the identity
    assert parallel_control_window(
        (1990, 1999),
        branch_year=1850,
        child_first_year=1850,
    ) == (1990, 1999)


def test_branch_year_in_parent() -> None:
    attributes = {
        "branch_time_in_parent": 10800.0,  # 30 years of a 360-day calendar
        "parent_time_units": "days since 1850-01-01",
    }
    cubes = _experiment(np.zeros(3), attributes=attributes)
    assert branch_year_in_parent(cubes) == 1880
    # Missing or undecodable metadata is reported as "unknown", not an error
    assert branch_year_in_parent(_experiment(np.zeros(3))) is None
    assert (
        branch_year_in_parent(
            _experiment(
                np.zeros(3),
                attributes={
                    "branch_time_in_parent": 0.0,
                    "parent_time_units": "not a time unit",
                },
            ),
        )
        is None
    )


# ---------------------------------------------------------------------------
# Fixed-SST forcing and the pattern effect
# ---------------------------------------------------------------------------


def test_amip_4xco2_erf() -> None:
    """A 7.5 W m-2 brighter TOA under quadrupled CO2 is the forcing."""
    diag = Amip4xCO2ERF(
        "amip_4xco2_erf",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    data = {
        "amip": _experiment(np.full(10, 0.5)),
        "amip4xco2": _experiment(np.full(10, 8.0)),
    }
    row = _row(diag.get_output(data, _info("amip")))
    assert list(row) == ["data_id", "data_type", "amip_4xco2_erf"]
    assert row["amip_4xco2_erf"] == pytest.approx(7.5, abs=1e-3)


def test_gfmip_patch_feedback() -> None:
    """East and west Pacific patches with prescribed feedbacks."""
    n_years = 10
    amip_flux = np.full(n_years, 0.5)
    amip_temperature = np.full(n_years, 288.0)
    # +1 K of patch warming, -2.0 and -0.8 W m-2 K-1 of feedback
    data = {
        "amip": _experiment(amip_flux, amip_temperature, temperature_name="ts"),
        "patch_ep": _experiment(
            amip_flux - 2.0,
            amip_temperature + 1.0,
            temperature_name="ts",
        ),
        "patch_wp": _experiment(
            amip_flux - 0.8,
            amip_temperature + 1.0,
            temperature_name="ts",
        ),
    }
    diag = GFMIPPatchFeedback(
        "gfmip_patch_feedback",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    row = _row(diag.get_output(data, _info("amip")))
    assert list(row) == [
        "data_id",
        "data_type",
        "patch_feedback_difference",
        "patch_feedback_ep",
        "patch_feedback_wp",
    ]
    assert row["patch_feedback_ep"] == pytest.approx(-2.0, abs=1e-3)
    assert row["patch_feedback_wp"] == pytest.approx(-0.8, abs=1e-3)
    assert row["patch_feedback_difference"] == pytest.approx(-1.2, abs=1e-3)


# ---------------------------------------------------------------------------
# Aerosol forcing
# ---------------------------------------------------------------------------

CONTROL_FIRST_YEAR = 1850
CONTROL_YEARS = 60
AEROSOL_YEARS = 30
EVALUATION_WINDOW = (1870, 1879)
BRANCH_YEAR = 1880
FEEDBACK = -1.2
AEROSOL_TEMPERATURE = 286.9
AEROSOL_FLUX = 0.05


def _control_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a drifting control whose parallel segment beats its mean."""
    years = CONTROL_FIRST_YEAR + np.arange(CONTROL_YEARS)
    temperature = 287.0 + 0.005 * np.arange(CONTROL_YEARS)
    flux = 0.1 + 0.001 * np.arange(CONTROL_YEARS)
    return years, temperature, flux


def _aerosol_data(*, with_branch_time: bool) -> dict[str, CubeList]:
    """piControl, abrupt-4xCO2 and hist-aer experiments with known answers."""
    _years, control_temperature, control_flux = _control_series()
    warming = 0.5 * np.arange(AEROSOL_YEARS)
    abrupt_temperature = control_temperature.mean() + warming
    abrupt_flux = 7.0 + FEEDBACK * warming
    attributes = (
        {
            "branch_time_in_parent": (BRANCH_YEAR - CONTROL_FIRST_YEAR) * DAYS_PER_YEAR,
            "parent_time_units": f"days since {CONTROL_FIRST_YEAR}-01-01",
        }
        if with_branch_time
        else None
    )
    return {
        "picontrol": _experiment(
            control_flux,
            control_temperature,
            first_year=CONTROL_FIRST_YEAR,
        ),
        "4xco2": _experiment(
            abrupt_flux,
            abrupt_temperature,
            first_year=CONTROL_FIRST_YEAR,
        ),
        "histaer": _experiment(
            np.full(AEROSOL_YEARS, AEROSOL_FLUX),
            np.full(AEROSOL_YEARS, AEROSOL_TEMPERATURE),
            first_year=CONTROL_FIRST_YEAR,
            attributes=attributes,
        ),
    }


def _expected_erf(baseline_first_year: int, baseline_last_year: int) -> float:
    """Get the forcing implied by a given control baseline."""
    years, temperature, flux = _control_series()
    in_baseline = (years >= baseline_first_year) & (years <= baseline_last_year)
    delta_temperature = AEROSOL_TEMPERATURE - temperature[in_baseline].mean()
    delta_flux = AEROSOL_FLUX - flux[in_baseline].mean()
    return delta_flux - FEEDBACK * delta_temperature


def test_aerosol_erf_uses_the_parallel_control_segment() -> None:
    diag = AerosolERF(
        "aerosol_erf",
        fail_on_missing_data=True,
        download_missing_data=False,
        evaluation_window=EVALUATION_WINDOW,
    )
    row = _row(diag.get_output(_aerosol_data(with_branch_time=True), _info()))
    assert list(row) == [
        "data_id",
        "data_type",
        "aerosol_erf",
        "aerosol_delta_n",
        "aerosol_delta_t",
        "aerosol_lambda",
        "aerosol_window_first_year",
        "aerosol_window_last_year",
        "aerosol_parallel_segment",
    ]
    assert row["aerosol_lambda"] == pytest.approx(FEEDBACK, abs=1e-6)
    assert row["aerosol_window_first_year"] == EVALUATION_WINDOW[0]
    assert row["aerosol_window_last_year"] == EVALUATION_WINDOW[1]
    assert row["aerosol_parallel_segment"] == 1.0
    # The child branched 30 years into the control, so the evaluation window
    # 1870-1879 runs alongside the control years 1900-1909
    assert row["aerosol_erf"] == pytest.approx(_expected_erf(1900, 1909), abs=2e-3)
    assert row["aerosol_delta_t"] < 0.0


def test_aerosol_erf_falls_back_to_the_long_term_control_mean() -> None:
    """Without branch metadata the control drift is not removed."""
    diag = AerosolERF(
        "aerosol_erf",
        fail_on_missing_data=True,
        download_missing_data=False,
        evaluation_window=EVALUATION_WINDOW,
    )
    row = _row(diag.get_output(_aerosol_data(with_branch_time=False), _info()))
    assert row["aerosol_parallel_segment"] == 0.0
    assert row["aerosol_erf"] == pytest.approx(
        _expected_erf(CONTROL_FIRST_YEAR, CONTROL_FIRST_YEAR + CONTROL_YEARS - 1),
        abs=2e-3,
    )
    # The two baselines give measurably different answers
    assert row["aerosol_erf"] != pytest.approx(_expected_erf(1900, 1909), abs=0.1)


def test_aerosol_erf_shifts_a_window_the_run_does_not_reach() -> None:
    """A run ending before the window keeps the window's length."""
    diag = AerosolERF(
        "aerosol_erf",
        fail_on_missing_data=True,
        download_missing_data=False,
        evaluation_window=(2010, 2019),
    )
    row = _row(diag.get_output(_aerosol_data(with_branch_time=True), _info()))
    last_year = CONTROL_FIRST_YEAR + AEROSOL_YEARS - 1
    assert row["aerosol_window_last_year"] == last_year
    assert row["aerosol_window_first_year"] == last_year - 9
