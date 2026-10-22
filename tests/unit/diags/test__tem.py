"""Tests for the Transformed Eulerian Mean (TEM) diagnostic.

The input fields are zonally uniform and given on the 2x2 degree target
latitudes and the PLEV19 levels, so preprocessing (level extraction,
regridding, zonal mean, climatology) leaves them untouched. Potential
temperature is chosen bilinear in latitude (radians) and pressure, which the
finite differences of the diagnostic differentiate exactly; the eddy flux is
chosen such that every TEM term has a closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import ComplexDataSource, DataSourceInformation
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags._utils import PLEV19_LEVELS
from climateeval.diags.complex import TEMDiagnostic
from climateeval.diags.complex._tem import (
    _CP,
    _EARTH_RADIUS,
    _G,
    _P0,
    _R_DRY,
    _make_zonal_clim_cube,
)

KAPPA = _R_DRY / _CP
P = np.array(PLEV19_LEVELS)  # Pa, surface first
P_TOP = P.min()
LAT = np.arange(-89.0, 90.0, 2.0)  # 2x2 degree target grid
PHI = np.deg2rad(LAT)
COS_PHI = np.cos(PHI)
LON = np.arange(0.0, 360.0, 30.0)

# theta = THETA_SURFACE + BETA phi + (GAMMA0 + GAMMA1 phi) (P0 - p)
THETA_SURFACE = 300.0  # K
BETA = -20.0  # K rad-1
GAMMA0 = 0.017  # K Pa-1
GAMMA1 = 0.004  # K Pa-1 rad-1; static stability weakest at the south pole
V_BAR = 2.0  # m s-1
OMEGA_BAR = -0.01  # Pa s-1
ALPHA = 0.5  # m s-1 rad-1; eddy_ratio = ALPHA p phi / cos(phi)

COLUMNS = [
    "data_id",
    "data_type",
    "air_pressure",
    "latitude",
    "psi_star",
    "v_star",
    "w_star",
    "w_dia",
]


def _info() -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthModel",
        category="model",
        institute="Synth",
        exp="amip",
        variant="r1i1p1f1",
    )


def _cube(var_name: str, units: str, field: np.ndarray) -> Cube:
    """Monthly (time, plev, lat, lon) cube, constant in time and longitude."""
    n_time = 2
    time = DimCoord(
        np.array([15.0, 45.0]),
        standard_name="time",
        units=Unit("days since 2000-01-01", calendar="360_day"),
    )
    plev = DimCoord(P, standard_name="air_pressure", units="Pa")
    lat = DimCoord(LAT, standard_name="latitude", units="degrees_north")
    lon = DimCoord(LON, standard_name="longitude", units="degrees_east", circular=True)
    for coord in (time, lat, lon):
        coord.guess_bounds()
    data = np.broadcast_to(
        field[None, :, :, None],
        (n_time, P.size, LAT.size, LON.size),
    )
    return Cube(
        data.astype(np.float32),
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (plev, 1), (lat, 2), (lon, 3)],
    )


def _stability() -> np.ndarray:
    """Analytic d(theta)/dp, shape (nlat,)."""
    return -(GAMMA0 + GAMMA1 * PHI)


def _exner() -> np.ndarray:
    """(p0 / p)^kappa, shape (nlev, 1)."""
    return ((_P0 / P) ** KAPPA)[:, None]


def _theta() -> np.ndarray:
    return (
        THETA_SURFACE
        + BETA * PHI[None, :]
        + (GAMMA0 + GAMMA1 * PHI[None, :]) * (_P0 - P[:, None])
    )


def _cubes(*, wap: bool = True, vt: bool = False) -> CubeList:
    shape = (P.size, LAT.size)
    cubes = CubeList(
        [
            _cube("va", "m s-1", np.full(shape, V_BAR)),
            _cube("ta", "K", _theta() / _exner()),
        ],
    )
    if wap:
        cubes.append(_cube("wap", "Pa s-1", np.full(shape, OMEGA_BAR)))
    if vt:
        # v'theta' = eddy_ratio * d(theta)/dp, and v'T' = v'theta' / Exner
        eddy_ratio = ALPHA * P[:, None] * PHI[None, :] / COS_PHI[None, :]
        cubes.append(
            _cube("vt", "K m s-1", eddy_ratio * _stability()[None, :] / _exner()),
        )
    return cubes


def _raw(cubes: CubeList) -> dict[str, np.ndarray]:
    diag = TEMDiagnostic("tem")
    raw = diag._calculate_raw_output(
        ComplexDataSource(data={"data": cubes}, information=_info()),
    )
    out: dict[str, np.ndarray] = {}
    for variable, cube in raw.items():
        assert cube.shape == (P.size, LAT.size)
        np.testing.assert_allclose(cube.coord("air_pressure").points, P)
        np.testing.assert_allclose(cube.coord("latitude").points, LAT)
        out[variable.var_name] = np.ma.filled(cube.data.astype(float), np.nan)
    return out


def _psi(v_star: np.ndarray) -> np.ndarray:
    """Streamfunction of a v* that is constant in pressure."""
    return (
        (2.0 * np.pi * _EARTH_RADIUS * COS_PHI[None, :] / _G)
        * v_star
        * (P[:, None] - P_TOP)
    )


def test_eulerian_mean_without_wap_and_vt() -> None:
    """Without omega and eddy flux: v* = v_bar, omega-based outputs are NaN."""
    out = _raw(_cubes(wap=False))
    np.testing.assert_allclose(out["v_star"], V_BAR, rtol=1e-6)
    expected_psi = _psi(np.full((1, LAT.size), V_BAR))
    np.testing.assert_allclose(out["psi_star"], expected_psi, rtol=1e-5, atol=1.0)
    # The streamfunction vanishes at the top and is northward-positive below
    np.testing.assert_allclose(out["psi_star"][-1], 0.0, atol=1.0)
    assert np.all(out["psi_star"][:-1] > 0.0)
    assert np.all(np.isnan(out["w_star"]))
    assert np.all(np.isnan(out["w_dia"]))


def test_diabatic_omega_from_thermodynamic_residual() -> None:
    """omega_dia = omega + v (d theta / a d phi) / (d theta / dp)."""
    out = _raw(_cubes())
    np.testing.assert_allclose(out["v_star"], V_BAR, rtol=1e-6)
    np.testing.assert_allclose(out["w_star"], OMEGA_BAR, rtol=1e-6)

    dtheta_dphi = BETA + GAMMA1 * (_P0 - P[:, None])
    expected = OMEGA_BAR + V_BAR * dtheta_dphi / (_EARTH_RADIUS * _stability()[None, :])
    finite = np.isfinite(out["w_dia"])
    np.testing.assert_allclose(out["w_dia"][finite], expected[finite], rtol=1e-3)
    # The horizontal advection term is a measurable correction to omega
    assert np.all(np.abs(expected - OMEGA_BAR) > 1e-4)

    # The weakest 1 % of the static stabilities are masked, and they all lie in
    # the southernmost column where d(theta)/dp is smallest
    n_masked = (~finite).sum()
    assert 0 < n_masked <= 0.02 * finite.size
    assert np.all(finite[:, 1:])


def test_full_tem_correction_with_eddy_heat_flux() -> None:
    """The eddy-induced circulation of a prescribed v'theta' / (d theta/dp)."""
    out = _raw(_cubes(vt=True))
    # v* = v_bar + d(eddy_ratio)/dp = v_bar + ALPHA phi / cos(phi)
    expected_v = V_BAR + ALPHA * PHI / COS_PHI
    np.testing.assert_allclose(
        out["v_star"],
        np.broadcast_to(expected_v, out["v_star"].shape),
        rtol=1e-3,
    )
    # omega* = omega_bar - d(cos(phi) eddy_ratio)/d phi / (a cos(phi))
    expected_w = OMEGA_BAR - ALPHA * P[:, None] / (_EARTH_RADIUS * COS_PHI[None, :])
    np.testing.assert_allclose(out["w_star"], expected_w, rtol=1e-3)
    # Psi* integrates v* from the model top
    np.testing.assert_allclose(
        out["psi_star"],
        _psi(expected_v[None, :]),
        rtol=1e-3,
        atol=1e3,
    )
    # The eddy term reverses the Eulerian circulation in the south
    assert np.all(out["v_star"][:, LAT < -75.0] < 0.0)
    # The diabatic velocity uses Eulerian means only: identical to no-vt case
    without_vt = _raw(_cubes())["w_dia"]
    np.testing.assert_allclose(out["w_dia"], without_vt, rtol=1e-6)


def test_get_output_merges_variables_into_one_table() -> None:
    diag = TEMDiagnostic("tem")
    output = diag.get_output({"data": _cubes()}, _info())
    assert output.raw_output is not None
    table = output.raw_output.to_pandas()
    assert list(table.columns) == COLUMNS
    # One row per (plev, lat) point, no NaN padding rows left over
    assert len(table) == P.size * LAT.size
    assert set(table["data_type"]) == {"to_benchmark"}
    assert table["v_star"].to_numpy() == pytest.approx(V_BAR, rel=1e-6)
    assert table["w_star"].to_numpy() == pytest.approx(OMEGA_BAR, rel=1e-6)
    assert table["psi_star"].notna().all()
    variables = set(output.variables.to_pandas()["id"])
    assert {"psi_star", "v_star", "w_star", "w_dia"} <= variables


def test_missing_required_key_raises() -> None:
    with pytest.raises(ValueError, match="Expected keys"):
        TEMDiagnostic("tem").get_output({"va": _cubes()}, _info())


def test_has_variable() -> None:
    cubes = _cubes(wap=False)
    assert TEMDiagnostic._has_variable(cubes, TEMDiagnostic._va)
    assert not TEMDiagnostic._has_variable(cubes, TEMDiagnostic._wap)
    assert not TEMDiagnostic._has_variable(cubes, TEMDiagnostic._vt)


def test_make_zonal_clim_cube() -> None:
    data = np.arange(P.size * LAT.size, dtype=float).reshape(P.size, LAT.size)
    cube = _make_zonal_clim_cube(data, P, LAT, "psi_star", "kg s-1")
    assert cube.var_name == "psi_star"
    assert cube.units == "kg s-1"
    assert cube.dtype == np.float32
    assert [c.name() for c in cube.dim_coords] == ["air_pressure", "latitude"]
    assert cube.coord("air_pressure").units == "Pa"
    np.testing.assert_array_equal(cube.data, data)


def test_postprocess_passes_through_empty_raw_output() -> None:
    output = DiagnosticOutput(
        raw_output=None,
        metrics=None,
        variables=None,  # type: ignore[arg-type]
        data_sources=None,  # type: ignore[arg-type]
    )
    assert TEMDiagnostic("tem")._postprocess_output(output) is output


def test_masked_points_do_not_enter_the_stability_threshold() -> None:
    """Masked (below-ground) values must not move the 1 % stability cut-off.

    The lowest levels south of 60S are masked, with a constant temperature
    underneath the mask. Its vertical gradient is near zero, so counting it
    would drop the cut-off to almost nothing; the unmasked points must then
    give the same omega_dia as without any mask.
    """
    reference = _raw(_cubes())
    cubes = _cubes()
    ta = cubes.extract_cube("ta")
    below_ground = np.zeros(ta.shape, dtype=bool)
    below_ground[:, :3, LAT < -60.0, :] = True
    ta.data = np.ma.masked_array(np.where(below_ground, 250.0, ta.data), below_ground)

    out = _raw(cubes)

    valid = np.isfinite(out["w_dia"]) & np.isfinite(reference["w_dia"])
    assert valid.sum() > 0.9 * valid.size
    np.testing.assert_allclose(out["w_dia"][valid], reference["w_dia"][valid])
    # The weak-stability mask of the unmasked run is kept
    assert np.all(~np.isfinite(out["w_dia"][~np.isfinite(reference["w_dia"])]))
