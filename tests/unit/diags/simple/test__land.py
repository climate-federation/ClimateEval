from __future__ import annotations

import numpy as np
import pytest
from iris.analysis.cartography import area_weights
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube, CubeList

from climateeval import Variable
from climateeval._utils import get_prepared_cube
from climateeval.diags.simple import (
    LandAnnualCycleSpatialDistribution,
    LandMap,
    LandTotalAnnualCycle,
    LandTotalTimeSeries,
    SimpleDiagnosticInputData,
)
from climateeval.diags.simple._land import (
    add_land_supplementaries,
    get_land_fraction,
    land_regrid,
    land_total,
)
from climateeval.exceptions import MissingDataError

GPP_STANDARD_NAME = "gross_primary_productivity_of_biomass_expressed_as_carbon"

#: Radius used by the regridder (iris default) and by e.g. CESM.
IRIS_RADIUS = 6367470.0
CESM_RADIUS = 6371220.0


def _grid(dlat: float, dlon: float) -> tuple[DimCoord, DimCoord]:
    lat = DimCoord(
        np.arange(-90.0 + dlat / 2, 90.0, dlat),
        standard_name="latitude",
        var_name="lat",
        units="degrees_north",
    )
    lon = DimCoord(
        np.arange(dlon / 2, 360.0, dlon),
        standard_name="longitude",
        var_name="lon",
        units="degrees_east",
    )
    lat.guess_bounds()
    lon.guess_bounds()
    return lat, lon


def _time(ntime: int) -> DimCoord:
    starts = np.arange(ntime) * 30.0
    return DimCoord(
        starts + 15.0,
        bounds=np.stack([starts, starts + 30.0], axis=1),
        standard_name="time",
        var_name="time",
        units="days since 2000-01-01",
    )


def _gpp_cube(data: np.ndarray, lat: DimCoord, lon: DimCoord) -> Cube:
    return Cube(
        data,
        standard_name=GPP_STANDARD_NAME,
        var_name="gpp",
        units="kg m-2 s-1",
        dim_coords_and_dims=[
            (_time(data.shape[0]), 0),
            (lat.copy(), 1),
            (lon.copy(), 2),
        ],
    )


def _fx_cube(data: np.ndarray, lat: DimCoord, lon: DimCoord, var_name: str) -> Cube:
    standard_name, units = {
        "sftlf": ("land_area_fraction", "%"),
        "areacella": ("cell_area", "m2"),
    }[var_name]
    return Cube(
        data,
        standard_name=standard_name,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(lat.copy(), 0), (lon.copy(), 1)],
    )


def _random_land(
    lat: DimCoord,
    lon: DimCoord,
    ntime: int = 3,
    seed: int = 0,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Get random GPP (masked over sea and one land cell) and sftlf (%)."""
    rng = np.random.default_rng(seed)
    shape = (lat.shape[0], lon.shape[0])
    sftlf = rng.uniform(0.0, 100.0, shape)
    sftlf[rng.random(shape) < 0.4] = 0.0  # sea
    sftlf[rng.random(shape) < 0.2] = 100.0  # interior land
    gpp = rng.uniform(1e-8, 5e-8, (ntime, *shape))
    mask = np.broadcast_to(sftlf == 0.0, gpp.shape).copy()
    mask[:, 3, 4] = True  # land without data (e.g. barren)
    return np.ma.masked_array(gpp, mask), sftlf


def _with_land(
    cube: Cube,
    sftlf: np.ndarray,
    cell_area: np.ndarray | None = None,
) -> Cube:
    cube = cube.copy()
    cube.add_ancillary_variable(
        AncillaryVariable(sftlf, standard_name="land_area_fraction", units="%"),
        (1, 2),
    )
    if cell_area is not None:
        cube.add_cell_measure(
            CellMeasure(
                cell_area,
                standard_name="cell_area",
                units="m2",
                measure="area",
            ),
            (1, 2),
        )
    return cube


def _expected_total(
    gpp: np.ma.MaskedArray,
    sftlf: np.ndarray,
    area: np.ndarray,
) -> np.ndarray:
    """Calculate sum(x * f * A) with f = 0 where x is masked."""
    fraction = np.where(np.ma.getmaskarray(gpp), 0.0, sftlf / 100.0)
    return (np.ma.filled(gpp, 0.0) * fraction * area).sum(axis=(-2, -1))


def _regridded_total(cube: Cube) -> np.ndarray:
    """Integrate the output of land_regrid with its land area weights."""
    land_area = cube.cell_measure("cell_area").data
    return (np.ma.filled(cube.data, 0.0) * land_area).sum(axis=(-2, -1))


@pytest.fixture
def land_cube() -> tuple[Cube, np.ma.MaskedArray, np.ndarray]:
    lat, lon = _grid(10.0, 10.0)
    gpp, sftlf = _random_land(lat, lon)
    return _with_land(_gpp_cube(gpp, lat, lon), sftlf), gpp, sftlf


# ----------------------------------------------------------------------------
# Land fraction and totals
# ----------------------------------------------------------------------------


def test_get_land_fraction(
    land_cube: tuple[Cube, np.ma.MaskedArray, np.ndarray],
) -> None:
    cube, gpp, sftlf = land_cube

    fraction = get_land_fraction(cube).compute()

    assert fraction.shape == cube.shape
    np.testing.assert_allclose(
        fraction,
        np.where(np.ma.getmaskarray(gpp), 0.0, sftlf / 100.0),
    )


def test_get_land_fraction_missing() -> None:
    lat, lon = _grid(10.0, 10.0)
    gpp, _ = _random_land(lat, lon)

    with pytest.raises(MissingDataError, match=r"No land_area_fraction \(sftlf\)"):
        get_land_fraction(_gpp_cube(gpp, lat, lon))


def test_land_total_area_from_bounds(
    land_cube: tuple[Cube, np.ma.MaskedArray, np.ndarray],
) -> None:
    cube, gpp, sftlf = land_cube

    total = land_total(cube)

    assert total.shape == (3,)
    assert total.units == "kg s-1"
    np.testing.assert_allclose(
        total.data,
        _expected_total(gpp, sftlf, area_weights(cube[0])),
        rtol=1e-12,
    )


def test_land_total_given_cell_area() -> None:
    lat, lon = _grid(10.0, 10.0)
    gpp, sftlf = _random_land(lat, lon)
    rng = np.random.default_rng(1)
    area = area_weights(_gpp_cube(gpp, lat, lon)[0]) * rng.uniform(0.9, 1.1)
    cube = _with_land(_gpp_cube(gpp, lat, lon), sftlf)
    cube.add_cell_measure(
        CellMeasure(area / 1e6, standard_name="cell_area", units="km2", measure="area"),
        (1, 2),
    )

    total = land_total(cube)

    np.testing.assert_allclose(
        total.data,
        _expected_total(gpp, sftlf, area),
        rtol=1e-12,
    )


def test_land_total_unstructured() -> None:
    lat, lon = _grid(30.0, 30.0)
    gpp, sftlf = _random_land(lat, lon, ntime=2)
    regular = _with_land(
        _gpp_cube(gpp, lat, lon),
        sftlf,
        area_weights(_gpp_cube(gpp, lat, lon)[0]),
    )
    cube = _to_unstructured(regular)

    total = land_total(cube)

    assert total.shape == (2,)
    np.testing.assert_allclose(total.data, land_total(regular).data, rtol=1e-12)


# ----------------------------------------------------------------------------
# Supplementaries of the data to benchmark
# ----------------------------------------------------------------------------


def _model_data() -> tuple[CubeList, np.ma.MaskedArray, np.ndarray, np.ndarray]:
    lat, lon = _grid(10.0, 10.0)
    gpp, sftlf = _random_land(lat, lon)
    cube = _gpp_cube(gpp, lat, lon)
    area = area_weights(cube[0])
    data = CubeList(
        [
            cube,
            _fx_cube(sftlf, lat, lon, "sftlf"),
            _fx_cube(area, lat, lon, "areacella"),
        ],
    )
    return data, gpp, sftlf, area


def test_add_land_supplementaries() -> None:
    data, gpp, sftlf, area = _model_data()
    variable = Variable("gpp", "gpp", "mon")

    cube = add_land_supplementaries(get_prepared_cube(data, variable), data, variable)

    assert cube.ancillary_variable_dims("land_area_fraction") == (1, 2)
    assert cube.cell_measure_dims("cell_area") == (1, 2)
    np.testing.assert_allclose(
        land_total(cube).data,
        _expected_total(gpp, sftlf, area),
        rtol=1e-12,
    )


def test_add_land_supplementaries_missing_areacella() -> None:
    data, *_ = _model_data()
    variable = Variable("gpp", "gpp", "mon")
    cube = get_prepared_cube(data, variable)

    with pytest.raises(MissingDataError, match="need 'areacella' in the input data"):
        add_land_supplementaries(cube, CubeList(data[:2]), variable)


def test_add_land_supplementaries_wrong_grid() -> None:
    data, *_ = _model_data()
    variable = Variable("gpp", "gpp", "mon")
    cube = get_prepared_cube(data, variable)
    lat, lon = _grid(20.0, 20.0)
    wrong = CubeList(
        [data[0], _fx_cube(np.full((9, 18), 50.0), lat, lon, "sftlf"), data[2]],
    )

    with pytest.raises(ValueError, match="does not match the grid of"):
        add_land_supplementaries(cube, wrong, variable)


# ----------------------------------------------------------------------------
# Conservative regridding
# ----------------------------------------------------------------------------


def test_land_regrid_coastal_cell() -> None:
    # One 20x20 degree target cell holds a coastal cell (f = 30 %, x = 10),
    # an interior land cell (f = 100 %, x = 2) and two sea cells
    lat, lon = _grid(10.0, 10.0)
    sftlf = np.zeros((18, 36))
    gpp = np.ma.masked_all((1, 18, 36))
    sftlf[10, 0], gpp[0, 10, 0] = 30.0, 10.0
    sftlf[11, 0], gpp[0, 11, 0] = 100.0, 2.0
    cube = _with_land(_gpp_cube(gpp, lat, lon), sftlf)
    area = area_weights(cube[0])
    coastal_area, interior_area = area[10, 0], area[11, 0]

    result = land_regrid(cube, "20x20")[0]

    lat_idx = int(np.argmin(np.abs(result.coord("latitude").points - 20.0)))
    lon_idx = int(np.argmin(np.abs(result.coord("longitude").points - 10.0)))
    expected = (10.0 * 0.3 * coastal_area + 2.0 * interior_area) / (
        0.3 * coastal_area + interior_area
    )
    np.testing.assert_allclose(result.data[lat_idx, lon_idx], expected, rtol=1e-6)
    np.testing.assert_allclose(
        result.cell_measure("cell_area").data[lat_idx, lon_idx],
        0.3 * coastal_area + interior_area,
        rtol=1e-6,
    )
    assert np.ma.count(result.data) == 1  # target cells with only sea are masked


@pytest.mark.parametrize(
    ("source", "target"),
    [((10.0, 10.0), "20x20"), ((7.5, 7.5), "2x2"), ((3.0, 4.0), "5x5")],
)
def test_land_regrid_conserves_total(
    source: tuple[float, float],
    target: str,
) -> None:
    lat, lon = _grid(*source)
    gpp, sftlf = _random_land(lat, lon)
    cube = _with_land(_gpp_cube(gpp, lat, lon), sftlf)
    native = land_total(cube).data

    result = land_regrid(cube, target)

    np.testing.assert_allclose(_regridded_total(result), native, rtol=1e-6)
    # The land area weights already include the land fraction (no double count)
    np.testing.assert_allclose(land_total(result).data, native, rtol=1e-6)
    np.testing.assert_allclose(
        result.ancillary_variable("land_area_fraction").data,
        100.0,
    )


def test_land_regrid_conserves_total_other_radius_masked_area() -> None:
    # Model cell areas on another Earth radius and masked over sea (like CLM)
    lat, lon = _grid(7.5, 7.5)
    gpp, sftlf = _random_land(lat, lon)
    area = area_weights(_gpp_cube(gpp, lat, lon)[0]) * (CESM_RADIUS / IRIS_RADIUS) ** 2
    area = np.ma.masked_where(sftlf == 0.0, area)
    cube = _with_land(_gpp_cube(gpp, lat, lon), sftlf, area)
    native = land_total(cube).data

    result = land_regrid(cube, "2x2")

    np.testing.assert_allclose(_regridded_total(result), native, rtol=1e-6)


def test_land_regrid_values_independent_of_grid_alignment() -> None:
    # A spatially constant field stays constant wherever there is land
    lat, lon = _grid(3.0, 4.0)
    gpp, sftlf = _random_land(lat, lon, ntime=1)
    gpp = np.ma.masked_array(np.full(gpp.shape, 3e-8), np.ma.getmaskarray(gpp))
    cube = _with_land(_gpp_cube(gpp, lat, lon), sftlf)

    result = land_regrid(cube, "5x5")

    np.testing.assert_allclose(result.data.compressed(), 3e-8, rtol=1e-6)


def _to_unstructured(cube: Cube) -> Cube:
    """Flatten a regular cube to an unstructured grid of quadrilateral cells."""
    lat = cube.coord("latitude")
    lon = cube.coord("longitude")
    lons, lats = np.meshgrid(lon.points, lat.points)
    lon_bounds = np.stack(
        np.broadcast_arrays(
            lon.bounds[None, :, 0],
            lon.bounds[None, :, 1],
            lon.bounds[None, :, 1],
            lon.bounds[None, :, 0],
        ),
        axis=-1,
    )
    lat_bounds = np.stack(
        np.broadcast_arrays(
            lat.bounds[:, None, 0],
            lat.bounds[:, None, 0],
            lat.bounds[:, None, 1],
            lat.bounds[:, None, 1],
        ),
        axis=-1,
    )
    ncells = lats.size
    lon_bounds = np.broadcast_to(lon_bounds, (*lats.shape, 4)).reshape(ncells, 4)
    lat_bounds = np.broadcast_to(lat_bounds, (*lats.shape, 4)).reshape(ncells, 4)
    data = cube.data.reshape(cube.shape[0], ncells)
    new = Cube(
        data,
        standard_name=cube.standard_name,
        var_name=cube.var_name,
        units=cube.units,
        dim_coords_and_dims=[(cube.coord("time").copy(), 0)],
        aux_coords_and_dims=[
            (
                AuxCoord(
                    lats.ravel(),
                    bounds=lat_bounds,
                    standard_name="latitude",
                    units="degrees_north",
                ),
                1,
            ),
            (
                AuxCoord(
                    lons.ravel(),
                    bounds=lon_bounds,
                    standard_name="longitude",
                    units="degrees_east",
                ),
                1,
            ),
        ],
    )
    new.add_ancillary_variable(
        AncillaryVariable(
            cube.ancillary_variable("land_area_fraction").data.ravel(),
            standard_name="land_area_fraction",
            units="%",
        ),
        1,
    )
    new.add_cell_measure(
        CellMeasure(
            np.ma.filled(cube.cell_measure("cell_area").data, 0.0).ravel(),
            standard_name="cell_area",
            units="m2",
            measure="area",
        ),
        1,
    )
    return new


def test_land_regrid_unstructured() -> None:
    lat, lon = _grid(10.0, 10.0)
    gpp, sftlf = _random_land(lat, lon, ntime=2)
    regular = _gpp_cube(gpp, lat, lon)
    cube = _to_unstructured(_with_land(regular, sftlf, area_weights(regular[0])))
    native = land_total(cube).data

    result = land_regrid(cube, "20x20")

    assert result.shape == (2, 9, 18)
    np.testing.assert_allclose(_regridded_total(result), native, rtol=1e-3)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


@pytest.fixture
def variable_gpp() -> Variable:
    return Variable("gpp", "gpp", "mon", units="g m-2 day-1")


@pytest.fixture
def variables_and_data(
    variable_gpp: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {variable_gpp: SimpleDiagnosticInputData(reference=None)}


def _year_of_land_data() -> Cube:
    lat, lon = _grid(10.0, 10.0)
    gpp, sftlf = _random_land(lat, lon, ntime=12)
    cube = _with_land(_gpp_cube(gpp, lat, lon), sftlf)
    cube.convert_units("g m-2 day-1")
    return cube


@pytest.mark.parametrize(
    ("diagnostic_cls", "shape", "units"),
    [
        (LandTotalAnnualCycle, (12,), "Pg yr-1"),
        (LandTotalTimeSeries, (1,), "Pg yr-1"),
        (LandMap, (90, 180), "g m-2 day-1"),
        (LandAnnualCycleSpatialDistribution, (12, 90, 180), "g m-2 day-1"),
    ],
)
def test__safely_preprocess(
    diagnostic_cls: type[LandMap],
    shape: tuple[int, ...],
    units: str,
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_gpp: Variable,
) -> None:
    diagnostic = diagnostic_cls("diagnostic", variables_and_data)

    cube = diagnostic._safely_preprocess(_year_of_land_data(), variable_gpp)

    assert cube.shape == shape
    assert cube.units == units


def test_land_total_annual_cycle_values(
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_gpp: Variable,
) -> None:
    cube = _year_of_land_data()
    diagnostic = LandTotalAnnualCycle("diagnostic", variables_and_data)

    result = diagnostic._safely_preprocess(cube, variable_gpp)

    # g day-1 -> Pg yr-1 (udunits year of 365.242198781 days)
    expected = land_total(cube).data * 365.242198781 / 1e15
    np.testing.assert_allclose(result.data, expected, rtol=1e-6)


@pytest.mark.parametrize("diagnostic_cls", [LandMap, LandTotalTimeSeries])
def test_missing_land_fraction_raises(
    diagnostic_cls: type[LandMap],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_gpp: Variable,
) -> None:
    cube = _year_of_land_data()
    cube.remove_ancillary_variable("land_area_fraction")
    diagnostic = diagnostic_cls("diagnostic", variables_and_data)

    with pytest.raises(MissingDataError, match="sftlf"):
        diagnostic._safely_preprocess(cube, variable_gpp)


@pytest.mark.parametrize("diagnostic_cls", [LandTotalAnnualCycle, LandTotalTimeSeries])
def test_land_total_variables_table_units(
    diagnostic_cls: type[LandTotalAnnualCycle],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_gpp: Variable,
) -> None:
    diagnostic = diagnostic_cls("diagnostic", variables_and_data)

    table = diagnostic._get_variables_table([variable_gpp]).to_pandas()

    assert table["units"].tolist() == ["Pg yr-1"]


def test_land_map_metrics_on_common_mask(
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_gpp: Variable,
) -> None:
    diagnostic = LandMap("diagnostic", variables_and_data)
    cube = diagnostic._safely_preprocess(_year_of_land_data(), variable_gpp)
    reference = cube.copy(data=cube.data * 0.9)
    # The model has data where the reference has none (e.g. barren land)
    first_valid = tuple(np.argwhere(~np.ma.getmaskarray(cube.data))[0])
    reference.data[first_valid] = np.ma.masked
    both_masked = cube.copy(
        data=np.ma.masked_where(np.ma.getmaskarray(reference.data), cube.data),
    )

    metrics = diagnostic._get_metrics(cube, reference)

    expected = diagnostic._get_metrics(both_masked, reference)
    for name, value in metrics.items():
        np.testing.assert_allclose(value, expected[name])
