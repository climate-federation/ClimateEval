from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from climateeval import Variable
from climateeval.diags.simple import (
    SeaIceAreaAnnualCycle,
    SeaIceAreaTimeSeries,
    SeaIceExtentAnnualCycle,
    SeaIceExtentTimeSeries,
    SimpleDiagnosticInputData,
)
from climateeval.diags.simple._sea_ice import SEA_ICE_EXTENT_THRESHOLD
from tests.sample_data.create_sample_data_sources import get_simple_2d_cube

if TYPE_CHECKING:
    from iris.cube import Cube

_AREA_DIAGNOSTICS = [SeaIceAreaAnnualCycle, SeaIceAreaTimeSeries]
_EXTENT_DIAGNOSTICS = [SeaIceExtentAnnualCycle, SeaIceExtentTimeSeries]
_DIAGNOSTICS = _AREA_DIAGNOSTICS + _EXTENT_DIAGNOSTICS


@pytest.fixture
def variable_siconc() -> Variable:
    return Variable("siconc", "siconc", "mon")


@pytest.fixture
def variables_and_data(
    variable_siconc: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {variable_siconc: SimpleDiagnosticInputData(reference=None)}


def _get_cube_siconc(concentration: float) -> Cube:
    """Get a cube with a spatially uniform sea-ice concentration in %."""
    cube = get_simple_2d_cube(
        var_name="siconc",
        standard_name="sea_ice_area_fraction",
        long_name="Sea-Ice Area Percentage (Ocean Grid)",
        units="%",
    )
    cube.data = np.full(cube.shape, concentration, dtype=np.float32)
    return cube


@pytest.mark.parametrize("diagnostic_cls", _DIAGNOSTICS)
def test_name(
    diagnostic_cls: type[SeaIceAreaAnnualCycle | SeaIceAreaTimeSeries],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
) -> None:
    assert diagnostic_cls("diagnostic", variables_and_data).name == "diagnostic"


@pytest.mark.parametrize("diagnostic_cls", _AREA_DIAGNOSTICS)
def test_area_has_no_threshold(
    diagnostic_cls: type[SeaIceAreaAnnualCycle | SeaIceAreaTimeSeries],
) -> None:
    assert diagnostic_cls._concentration_threshold is None


@pytest.mark.parametrize("diagnostic_cls", _EXTENT_DIAGNOSTICS)
def test_extent_threshold(
    diagnostic_cls: type[SeaIceExtentAnnualCycle | SeaIceExtentTimeSeries],
) -> None:
    assert diagnostic_cls._concentration_threshold == SEA_ICE_EXTENT_THRESHOLD
    assert SEA_ICE_EXTENT_THRESHOLD == 15.0


@pytest.mark.parametrize("diagnostic_cls", _DIAGNOSTICS)
def test__safely_preprocess(
    diagnostic_cls: type[SeaIceAreaAnnualCycle | SeaIceAreaTimeSeries],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_siconc: Variable,
) -> None:
    diagnostic = diagnostic_cls("diagnostic", variables_and_data)

    cube = diagnostic._safely_preprocess(_get_cube_siconc(50.0), variable_siconc)

    assert cube.units == "1e6 km2"


@pytest.mark.parametrize(
    ("area_cls", "extent_cls"),
    zip(_AREA_DIAGNOSTICS, _EXTENT_DIAGNOSTICS, strict=True),
)
def test_extent_counts_ice_covered_cells_fully(
    area_cls: type[SeaIceAreaAnnualCycle | SeaIceAreaTimeSeries],
    extent_cls: type[SeaIceExtentAnnualCycle | SeaIceExtentTimeSeries],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_siconc: Variable,
) -> None:
    # Every cell is half covered, i.e. above the 15 % threshold: the extent
    # counts them fully and is therefore exactly twice the area
    cube_siconc = _get_cube_siconc(50.0)

    area = area_cls("diagnostic", variables_and_data)._safely_preprocess(
        cube_siconc,
        variable_siconc,
    )
    extent = extent_cls("diagnostic", variables_and_data)._safely_preprocess(
        cube_siconc,
        variable_siconc,
    )

    assert np.all(area.data > 0.0)
    np.testing.assert_allclose(extent.data, 2.0 * area.data, rtol=1e-5)


@pytest.mark.parametrize(
    ("area_cls", "extent_cls"),
    zip(_AREA_DIAGNOSTICS, _EXTENT_DIAGNOSTICS, strict=True),
)
def test_extent_ignores_masked_cells(
    area_cls: type[SeaIceAreaAnnualCycle | SeaIceAreaTimeSeries],
    extent_cls: type[SeaIceExtentAnnualCycle | SeaIceExtentTimeSeries],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_siconc: Variable,
) -> None:
    # Cells without data (e.g. land) must not count towards the extent
    cube_siconc = _get_cube_siconc(100.0)
    cube_masked = cube_siconc.copy()
    mask = np.zeros(cube_masked.shape, dtype=bool)
    mask[:, 0, :] = True
    cube_masked.data = np.ma.masked_array(cube_masked.data, mask=mask)

    diagnostic = extent_cls("diagnostic", variables_and_data)
    extent = diagnostic._safely_preprocess(cube_siconc, variable_siconc)
    masked_extent = diagnostic._safely_preprocess(cube_masked, variable_siconc)
    masked_area = area_cls("diagnostic", variables_and_data)._safely_preprocess(
        cube_masked,
        variable_siconc,
    )

    assert np.all(masked_extent.data < extent.data)
    assert np.all(masked_extent.data > 0.0)
    # On a fully ice-covered field, extent and area agree, so the masked cells
    # are excluded from both in exactly the same way
    np.testing.assert_allclose(masked_extent.data, masked_area.data, rtol=1e-5)


@pytest.mark.parametrize(
    ("area_cls", "extent_cls"),
    zip(_AREA_DIAGNOSTICS, _EXTENT_DIAGNOSTICS, strict=True),
)
def test_extent_ignores_cells_below_threshold(
    area_cls: type[SeaIceAreaAnnualCycle | SeaIceAreaTimeSeries],
    extent_cls: type[SeaIceExtentAnnualCycle | SeaIceExtentTimeSeries],
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    variable_siconc: Variable,
) -> None:
    # Marginal ice below the threshold contributes to the area, but not to the
    # extent
    cube_siconc = _get_cube_siconc(10.0)

    area = area_cls("diagnostic", variables_and_data)._safely_preprocess(
        cube_siconc,
        variable_siconc,
    )
    extent = extent_cls("diagnostic", variables_and_data)._safely_preprocess(
        cube_siconc,
        variable_siconc,
    )

    assert np.all(area.data > 0.0)
    np.testing.assert_allclose(extent.data, 0.0, atol=1e-10)
