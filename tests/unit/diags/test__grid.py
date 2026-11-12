"""Test the ``grid`` diagnostic kwarg (CE-9)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iris.cube import CubeList

from climateeval.diags._utils import DEFAULT_GRID, SEA_ICE_GRID
from climateeval.diags.complex import ECS, Amip4xCO2ERF
from climateeval.diags.simple import (
    AMOCTimeSeries,
    AnnualCycle,
    Map,
    SeaIceAreaAnnualCycle,
    SeaIceExtentTimeSeries,
    SimpleDiagnosticInputData,
)

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


def _lat_lon_shape(cube: Cube) -> tuple[int, int]:
    return (
        cube.coord("latitude").shape[0],
        cube.coord("longitude").shape[0],
    )


def _map(variable: Variable, **kwargs: str) -> Map:
    return Map("map", {variable: SimpleDiagnosticInputData(reference=None)}, **kwargs)


def test_defaults() -> None:
    assert DEFAULT_GRID == "2x2"
    assert SEA_ICE_GRID == "1x1"


def test_simple_default_grid(cube_2d: Cube, variable_tas: Variable) -> None:
    diagnostic = _map(variable_tas)
    assert diagnostic._target_grid == DEFAULT_GRID
    cube = diagnostic._safely_preprocess(cube_2d, variable_tas)
    assert _lat_lon_shape(cube) == (90, 180)


def test_simple_grid_kwarg(cube_2d: Cube, variable_tas: Variable) -> None:
    diagnostic = _map(variable_tas, grid="1x1")
    cube = diagnostic._safely_preprocess(cube_2d, variable_tas)
    assert _lat_lon_shape(cube) == (180, 360)


def test_simple_grid_other_spec(cube_2d: Cube, variable_tas: Variable) -> None:
    diagnostic = _map(variable_tas, grid="2.5x2.5")
    cube = diagnostic._safely_preprocess(cube_2d, variable_tas)
    assert _lat_lon_shape(cube) == (72, 144)


def test_simple_grid_annual_cycle(cube_2d: Cube, variable_tas: Variable) -> None:
    # Non-map diagnostics use the grid too (area mean, so check it runs)
    data = {variable_tas: SimpleDiagnosticInputData(reference=None)}
    assert AnnualCycle("ac", data, grid="1x1")._target_grid == "1x1"
    assert AnnualCycle("ac", data)._target_grid == DEFAULT_GRID
    AnnualCycle("ac", data, grid="1x1")._safely_preprocess(cube_2d, variable_tas)


@pytest.mark.parametrize("cls", [SeaIceAreaAnnualCycle, SeaIceExtentTimeSeries])
def test_sea_ice_default_grid(cls: type, variable_tas: Variable) -> None:
    data = {variable_tas: SimpleDiagnosticInputData(reference=None)}
    assert cls("si", data)._target_grid == SEA_ICE_GRID
    assert cls("si", data, grid="2x2")._target_grid == "2x2"


def test_diagnostic_that_does_not_regrid_accepts_grid(variable_tas: Variable) -> None:
    data = {variable_tas: SimpleDiagnosticInputData(reference=None)}
    assert AMOCTimeSeries("amoc", data, grid="1x1")._target_grid == "1x1"


def test_complex_grid_kwarg() -> None:
    assert ECS("ecs")._target_grid == DEFAULT_GRID
    assert ECS("ecs", grid="1x1")._target_grid == "1x1"
    assert "grid" not in ECS("ecs", grid="1x1")._kwargs


def test_complex_scalar_regrids_to_grid(cube_2d: Cube, variable_tas: Variable) -> None:
    default = Amip4xCO2ERF("erf")._prepared_cube(CubeList([cube_2d]), variable_tas)
    assert _lat_lon_shape(default) == (90, 180)
    fine = Amip4xCO2ERF("erf", grid="1x1")._prepared_cube(
        CubeList([cube_2d]),
        variable_tas,
    )
    assert _lat_lon_shape(fine) == (180, 360)


def test_subclass_skipping_init_uses_default_grid() -> None:
    # Subclasses (and tests) that bypass __init__ still regrid to the default
    assert Map.__new__(Map)._target_grid == DEFAULT_GRID
    assert SeaIceAreaAnnualCycle.__new__(SeaIceAreaAnnualCycle)._target_grid == (
        SEA_ICE_GRID
    )
