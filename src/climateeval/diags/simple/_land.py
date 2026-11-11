"""Provide land weighting helpers and the base class of land diagnostics.

Land fluxes (e.g. ``gpp``) are given per unit land area of a grid cell, so
every horizontal operation needs the land fraction ``f`` (``sftlf``) and the
cell area ``A`` (``areacella``):

* Totals are computed on the native grid as ``T = sum(x * f * A)``.
* Maps are regridded conservatively as ``R(x * f) / R(f)``, so that coastal
  cells only contribute their land part. Their metrics are weighted by the
  land area ``A * R(f)`` of the target grid.

Cells where ``x`` is masked get ``f = 0``, so they count neither in the totals
nor in the regridded means. Data without a land fraction is skipped.

"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

import dask.array as da
import ibis
import iris.analysis
from esmvalcore.iris_helpers import has_unstructured_grid
from esmvalcore.preprocessor import (
    annual_statistics,
    climate_statistics,
    convert_units,
    regrid,
    regrid_time,
)
from esmvalcore.preprocessor._shared import try_adding_calculated_cell_area
from iris.coords import AncillaryVariable, CellMeasure, CellMethod
from iris.exceptions import ConstraintMismatchError
from iris.mesh import MeshXY
from iris.util import broadcast_to_shape
from iris.warnings import IrisUserWarning
from loguru import logger

from climateeval import Coordinate, Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import get_prepared_cube
from climateeval._variable import COORDINATES
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic
from climateeval.exceptions import MissingDataError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    from iris.cube import Cube, CubeList
    from xarray import Dataset

logger = logger.opt(colors=True)

LAND_FRACTION = "land_area_fraction"
CELL_AREA = "cell_area"
HORIZONTAL_COORDS = ("latitude", "longitude")

#: Target units of global land totals (e.g. PgC yr-1 for ``gpp``).
_LAND_TOTAL_UNITS = "Pg yr-1"


def _horizontal_dims(cube: Cube) -> tuple[int, ...]:
    """Get the (sorted) dimensions spanned by latitude and longitude."""
    return tuple(
        sorted(set(cube.coord_dims("latitude") + cube.coord_dims("longitude"))),
    )


def _horizontal_coords(cube: Cube) -> list[str]:
    """Get coordinates to collapse the horizontal dimensions.

    On unstructured or curvilinear grids, latitude and longitude span the same
    dimensions, so collapsing latitude alone collapses them.
    """
    if cube.coord_dims("latitude") == cube.coord_dims("longitude"):
        return ["latitude"]
    return list(HORIZONTAL_COORDS)


def _horizontal_slice(cube: Cube) -> Cube:
    """Get the first horizontal slice of a cube."""
    return next(cube.slices(_horizontal_dims(cube), ordered=False))


def _without_supplementaries_on_mesh(cube: Cube) -> Cube:
    """Copy a cube without supplementaries, on a mesh if it is unstructured.

    Conservative regridding of unstructured grids needs a mesh, which is built
    from the cell bounds of latitude and longitude.
    """
    cube = _without_supplementaries(cube)
    if cube.mesh is not None or not has_unstructured_grid(cube):
        return cube
    lat = cube.coord("latitude")
    lon = cube.coord("longitude")
    if not (lat.has_bounds() and lon.has_bounds()):
        msg = (
            f"Conservative regridding of the unstructured grid of "
            f"{cube.summary(shorten=True)} needs latitude and longitude bounds"
        )
        raise ValueError(msg)
    mesh = MeshXY.from_coords(lon, lat)
    dim = cube.coord_dims(lat)[0]
    cube.remove_coord(lat)
    cube.remove_coord(lon)
    for mesh_coord in mesh.to_MeshCoords("face"):
        cube.add_aux_coord(mesh_coord, dim)
    return cube


def _check_same_grid(cube: Cube, field: Cube) -> tuple[int, ...]:
    """Check that a horizontal field lies on the grid of a cube."""
    dims = _horizontal_dims(cube)
    if field.shape != tuple(cube.shape[d] for d in dims):
        msg = (
            f"{field.name()} (shape {field.shape}) does not match the grid of "
            f"{cube.summary(shorten=True)}"
        )
        raise ValueError(msg)
    return dims


def add_land_supplementaries(
    cube: Cube,
    data: CubeList | Dataset,
    variable: Variable,
) -> Cube:
    """Attach ``sftlf`` and ``areacella`` from the input data to a cube.

    Both are prepared with the regional settings of ``variable`` so that they
    end up on exactly the grid of ``cube``.

    Raises
    ------
    MissingDataError
        ``sftlf`` or ``areacella`` is not part of the input data.

    """
    cube = cube.copy()
    for var_name in ("sftlf", "areacella"):
        fx_variable = Variable(
            var_name,
            var_name,
            "fx",
            min_lat=variable.min_lat,
            max_lat=variable.max_lat,
            min_lon=variable.min_lon,
            max_lon=variable.max_lon,
        )
        try:
            field = get_prepared_cube(data, fx_variable)
        except ConstraintMismatchError as exc:
            msg = (
                f"Land diagnostics need '{var_name}' in the input data to "
                f"weight {cube.summary(shorten=True)}"
            )
            raise MissingDataError(msg) from exc
        dims = _check_same_grid(cube, field)
        if var_name == "sftlf":
            if cube.ancillary_variables(LAND_FRACTION):
                cube.remove_ancillary_variable(LAND_FRACTION)
            cube.add_ancillary_variable(
                AncillaryVariable(
                    field.core_data(),
                    standard_name=field.standard_name,
                    long_name=field.long_name,
                    var_name=field.var_name,
                    units=field.units,
                ),
                dims,
            )
        else:
            if cube.cell_measures(CELL_AREA):
                cube.remove_cell_measure(CELL_AREA)
            cube.add_cell_measure(
                CellMeasure(
                    field.core_data(),
                    standard_name=field.standard_name,
                    long_name=field.long_name,
                    var_name=field.var_name,
                    units=field.units,
                    measure="area",
                ),
                dims,
            )
    return cube


def get_land_fraction(cube: Cube) -> da.Array:
    """Get the land fraction (in [0, 1]) where the data is valid.

    The ``land_area_fraction`` ancillary variable of the cube is used. The
    result has the shape of the cube and is 0 where the data is masked.

    Raises
    ------
    MissingDataError
        The cube has no land fraction (so the dataset is skipped).

    """
    if not cube.ancillary_variables(LAND_FRACTION):
        msg = f"No {LAND_FRACTION} (sftlf) available for {cube.summary(shorten=True)}"
        raise MissingDataError(msg)
    ancillary = cube.ancillary_variable(LAND_FRACTION).copy()
    ancillary.convert_units("1")
    fraction = broadcast_to_shape(
        da.asarray(ancillary.core_data()),
        cube.shape,
        cube.ancillary_variable_dims(ancillary.name()),
    )
    valid = ~da.ma.getmaskarray(da.asarray(cube.core_data()))
    return da.where(valid, da.ma.filled(fraction, 0.0), 0.0)


def get_cell_area(cube: Cube) -> da.Array:
    """Get the cell area (in m2) broadcast to the shape of the cube.

    The ``cell_area`` cell measure of the cube is used; without one, the area
    is calculated from the coordinate bounds (regular grids only).

    """
    cube = cube.copy()
    with setup_esmvaltool_config_and_logging():
        try_adding_calculated_cell_area(cube)
    measure = cube.cell_measure(CELL_AREA).copy()
    measure.convert_units("m2")
    return broadcast_to_shape(
        da.asarray(measure.core_data()),
        cube.shape,
        cube.cell_measure_dims(CELL_AREA),
    )


def _without_supplementaries(cube: Cube) -> Cube:
    """Copy a cube without ancillary variables and cell measures."""
    cube = cube.copy()
    for ancillary in cube.ancillary_variables():
        cube.remove_ancillary_variable(ancillary)
    for measure in cube.cell_measures():
        cube.remove_cell_measure(measure)
    return cube


def land_total(cube: Cube) -> Cube:
    """Integrate a per-land-area quantity over land on its native grid.

    Calculates ``T = sum(x * f * A)`` over latitude and longitude, with ``f``
    from :func:`get_land_fraction` and ``A`` from :func:`get_cell_area`. The
    units of the result are those of ``x`` times m2 (e.g. kg s-1 for
    ``gpp``).

    """
    weights = get_land_fraction(cube) * get_cell_area(cube)
    data = da.ma.filled(da.asarray(cube.core_data()), 0.0) * weights
    product = _without_supplementaries(cube).copy(data=data)
    product.units = cube.units * "m2"
    with warnings.catch_warnings():
        # The data is already weighted by the land area
        warnings.filterwarnings(
            "ignore",
            message=r"Collapsing spatial coordinate .* without weighting",
            category=IrisUserWarning,
            module="iris",
        )
        total = product.collapsed(_horizontal_coords(product), iris.analysis.SUM)
    total.cell_methods = (
        *cube.cell_methods,
        CellMethod("sum", HORIZONTAL_COORDS, comments="over land"),
    )
    return total


def land_regrid(
    cube: Cube,
    target_grid: str | Cube,
    *,
    min_fraction: float = 0.0,
) -> Cube:
    """Regrid a per-land-area quantity conservatively, weighting by land.

    Calculates ``R(x * f) / R(f)`` with the conservative regridder ``R``
    (``area_weighted``, which uses ESMF for irregular grids and meshes;
    unstructured grids are turned into a mesh first). Target cells with
    ``R(f) <= min_fraction`` are masked.

    The result carries the land area ``A * R(f)`` (in m2, averaged over any
    non-horizontal dimensions) as ``cell_area``, so that ESMValCore's weighted
    metrics weight by land area. Since that area is land only, its
    ``land_area_fraction`` is 100 %, which keeps :func:`land_total` of the
    result equal to the total on the native grid. The land area is scaled so
    that the target grid covers the same area as the source grid, which
    matters when the source cell areas (e.g. a model's ``areacella``) use a
    different Earth radius than the regridder.

    """
    fraction = get_land_fraction(cube)
    clean = _without_supplementaries_on_mesh(cube)
    flux = clean.copy(data=da.ma.filled(da.asarray(cube.core_data()), 0.0) * fraction)
    land = clean.copy(data=fraction)
    land.units = "1"
    source_slice = _horizontal_slice(cube)
    source_area = get_cell_area(source_slice)  # may be masked (e.g. over sea)
    has_area = ~da.ma.getmaskarray(source_area)
    coverage = _without_supplementaries_on_mesh(source_slice)
    coverage = coverage.copy(data=has_area.astype(float))

    with setup_esmvaltool_config_and_logging():
        flux = regrid(flux, target_grid, "area_weighted", cache_weights=True)
        land = regrid(land, target_grid, "area_weighted", cache_weights=True)
        coverage = regrid(coverage, target_grid, "area_weighted", cache_weights=True)

    regridded_fraction = da.asarray(land.core_data())
    regridded_fraction = da.ma.filled(regridded_fraction, 0.0)
    data = da.ma.masked_where(
        regridded_fraction <= min_fraction,
        da.ma.filled(da.asarray(flux.core_data()), 0.0)
        / da.where(regridded_fraction > min_fraction, regridded_fraction, 1.0),
    )
    result = flux.copy(data=data)
    result.units = cube.units

    dims = _horizontal_dims(result)
    other_dims = tuple(d for d in range(result.ndim) if d not in dims)
    static_fraction = regridded_fraction
    if other_dims:
        static_fraction = static_fraction.mean(axis=other_dims)
    result.add_ancillary_variable(
        AncillaryVariable(
            da.full_like(static_fraction, 100.0),
            standard_name=LAND_FRACTION,
            var_name="sftlf",
            units="%",
        ),
        dims,
    )
    # Target areas consistent with the source areas: the target area covered
    # by source cells with an area, sum(A_target * R(has_area)), must equal
    # their source area
    area = get_cell_area(_horizontal_slice(result))
    covered_area = (area * da.ma.filled(da.asarray(coverage.core_data()), 0.0)).sum()
    area = area * (da.ma.filled(source_area, 0.0).sum() / covered_area)
    result.add_cell_measure(
        CellMeasure(
            area * static_fraction,
            standard_name=CELL_AREA,
            var_name="areacella",
            units="m2",
            measure="area",
        ),
        dims,
    )
    return result


class LandDiagnostic(SimpleDiagnostic):
    """Base class of land diagnostics.

    The data to benchmark must provide ``sftlf`` and ``areacella``, which are
    attached to every variable before preprocessing. Reference and other data
    sources provide their own land fraction (e.g.
    :class:`climateeval.data.FLUXCOM`, or ``sftlf`` downloaded with CMIP6
    data). Data sources without one are skipped with a warning (see
    :func:`get_land_fraction`).

    """

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube with its land fraction and cell area."""
        cube = get_prepared_cube(data, variable)
        cube = add_land_supplementaries(cube, data, variable)
        return self._safely_preprocess(cube, variable)


class _LandTotalDiagnostic(LandDiagnostic):
    """Base class of land total diagnostics (output in ``Pg yr-1``)."""

    def _get_variables_table(self, variables: Iterable[Variable]) -> ibis.Table:
        """Get variables table with the units of the land totals."""
        return (
            super()
            ._get_variables_table(variables)
            .mutate(units=ibis.literal(_LAND_TOTAL_UNITS))
        )


class LandTotalAnnualCycle(_LandTotalDiagnostic):
    """Manage land total annual cycle diagnostics.

    Computes the monthly climatology of the land total (see
    :func:`land_total`) on the native grid, in ``Pg yr-1``.
    """

    _final_coordinates = (Coordinate("month_number"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = convert_units(land_total(cube), _LAND_TOTAL_UNITS)
            cube = climate_statistics(cube, "mean", "month")
        return cube  # noqa: RET504


class LandTotalTimeSeries(_LandTotalDiagnostic):
    """Manage land total annual mean time series diagnostics.

    Computes the annual mean of the land total (see :func:`land_total`) on the
    native grid, in ``Pg yr-1``.
    """

    _final_coordinates = (Coordinate("time"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = convert_units(land_total(cube), _LAND_TOTAL_UNITS)
            cube = annual_statistics(cube, "mean")
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504


class _LandSpatialDiagnostic(LandDiagnostic):
    """Base class of land diagnostics on the common grid.

    The climatology is calculated on the native grid and then regridded with
    :func:`land_regrid`, so the metrics are weighted by land area. They are
    calculated on the cells where both datasets have data.
    """

    #: Period of the climatology (see ``climate_statistics``).
    _period: ClassVar[str]

    def _get_metrics(self, cube: Cube, reference_cube: Cube) -> dict[str, pd.Series]:
        """Get metrics on the common mask of both datasets."""
        mask = da.ma.getmaskarray(da.asarray(cube.core_data())) | da.ma.getmaskarray(
            da.asarray(reference_cube.core_data()),
        )
        cube = cube.copy(data=da.ma.masked_where(mask, cube.core_data()))
        reference_cube = reference_cube.copy(
            data=da.ma.masked_where(mask, reference_cube.core_data()),
        )
        return super()._get_metrics(cube, reference_cube)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = climate_statistics(cube, "mean", self._period)
        return land_regrid(cube, DEFAULT_GRID)


class LandMap(_LandSpatialDiagnostic):
    """Manage land map diagnostics.

    Computes the time-mean map on the common grid, regridded conservatively
    with the land fraction (see :func:`land_regrid`).
    """

    _final_coordinates = (Coordinate("lat"), Coordinate("lon"))

    _period = "full"


class LandAnnualCycleSpatialDistribution(_LandSpatialDiagnostic):
    """Manage land annual cycle spatial distribution diagnostics.

    Computes the monthly climatology on the common grid, regridded
    conservatively with the land fraction (see :func:`land_regrid`). Only
    metrics are output.
    """

    _final_coordinates = (
        Coordinate("month_number"),
        Coordinate("lat"),
        Coordinate("lon"),
    )

    _output_data_table = False

    _period = "month"
