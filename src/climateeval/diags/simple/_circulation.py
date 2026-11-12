"""Provide atmospheric circulation index diagnostics.

Three generic indices, each computed from a single variable so that the same
code runs on a model, on a reanalysis and on any comparison data set through
the standard :class:`SimpleDiagnostic` machinery:

- :class:`HadleyCellEdge`: latitude where the 500 hPa mass streamfunction of
  the zonal-mean meridional wind ``va`` crosses zero poleward of its
  subtropical extremum (TropD ``PSI_500``; Adam et al., 2018, GMD).
- :class:`EddyDrivenJetLatitude`: latitude of the maximum of the zonal-mean
  850 hPa zonal wind ``ua`` between 20 and 70 degrees, refined by a parabola
  (TropD ``EDJ``; Barnes and Polvani, 2013).
- :class:`WalkerCirculationStrength`: sea-level pressure of the central-east
  Pacific minus that of the Indian Ocean-west Pacific (Vecchi et al., 2006).

The index is computed from the *mean* field of each period (the annual or
seasonal mean ``va`` or ``ua``), as in TropD, and is written as a time series
with one value per period. Options are given per variable through the
``diagnostic_kwargs`` of the variable definition:

``aggregation``
    ``"annual"`` (default), ``"monthly"`` or a season (``"DJF"``, ``"MAM"``,
    ``"JJA"``, ``"SON"``): one value per year (season) or per month.
``grid``
    Target grid of the regridding (default :data:`DEFAULT_GRID`).

Longitude sectors (basin jets, Hadley-cell edge of one sector) are selected
with the variable's ``min_lon``/``max_lon``.

Output cube coordinates: (time,).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    annual_statistics,
    extract_levels,
    regrid,
    regrid_time,
    seasonal_statistics,
)
from iris.cube import Cube

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags import _physics
from climateeval.diags._cube_arrays import cube_to_array, nan_mean
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from climateeval import Variable

_SEASONS = ("DJF", "MAM", "JJA", "SON")
_SEASON_CENTRE_MONTH = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}
_PRESSURE_COORD = "air_pressure"


def _aggregate_time(cube: Cube, aggregation: str) -> Cube:
    """Average a cube to the period of the index."""
    if aggregation == "monthly":
        return cube
    if aggregation == "annual":
        return annual_statistics(cube, "mean")
    if aggregation.upper() in _SEASONS:
        # The seasons have to partition the year, so average all four and keep
        # the requested one, identified by the central month of its time point
        # (iris drops incomplete seasons, e.g. a leading Jan-Feb without its
        # December).
        season = aggregation.upper()
        cube = seasonal_statistics(cube, "mean", seasons=_SEASONS)
        coord = cube.coord("time")
        months = np.array([d.month for d in coord.units.num2date(coord.points)])
        selected = np.flatnonzero(months == _SEASON_CENTRE_MONTH[season])
        if selected.size == 0:
            msg = f"no complete {season} season in the data"
            raise ValueError(msg)
        return cube[selected]
    msg = (
        f"'aggregation' must be 'annual', 'monthly' or one of {_SEASONS}, "
        f"got '{aggregation}'"
    )
    raise ValueError(msg)


class _CirculationIndex(SimpleDiagnostic):
    """Manage diagnostics reducing a field to one index value per period."""

    _final_coordinates = (Coordinate("time"),)

    _index_units: ClassVar[str] = "1"
    """Units of the index."""

    _index_name: ClassVar[str] = "circulation_index"
    """Name of the output cube (replaced by the variable ID in the table)."""

    def _preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Preprocess data to an index time series."""
        options = variable.diagnostic_kwargs
        aggregation = str(options.get("aggregation", "annual"))
        with setup_esmvaltool_config_and_logging():
            cube = self._prepare_field(cube, variable)
            cube = regrid(
                cube,
                options.get("grid", DEFAULT_GRID),
                "linear",
                cache_weights=True,
            )
            cube = _aggregate_time(cube, aggregation)
            values = self._index_values(cube, options)
            time_coord = cube.coord("time").copy()
            index = Cube(
                values.astype(float),
                var_name=self._index_name,
                units=self._output_units(cube),
                dim_coords_and_dims=[(time_coord, 0)],
            )
            return regrid_time(
                index,
                frequency="mon" if aggregation == "monthly" else "yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )

    def _prepare_field(self, cube: Cube, variable: Variable) -> Cube:  # noqa: ARG002
        """Select the levels needed before regridding (default: nothing)."""
        return cube

    def _output_units(self, cube: Cube) -> str:  # noqa: ARG002
        """Get the units of the index."""
        return self._index_units

    def _index_values(self, cube: Cube, options: dict[str, Any]) -> np.ndarray:
        """Compute the index for every time step of an aggregated cube."""
        raise NotImplementedError


class HadleyCellEdge(_CirculationIndex):
    """Manage Hadley-cell edge diagnostics (500 hPa streamfunction zero).

    Input: monthly ``va`` on pressure levels covering at least the troposphere
    up to 500 hPa. The zonal-mean ``va`` of each period gives the mass
    streamfunction (:func:`~climateeval.diags._physics.meridional_streamfunction`)
    whose zero crossing poleward of its subtropical extremum is the edge
    (:func:`~climateeval.diags._physics.hadley_cell_edge`), in degrees
    (negative for the southern hemisphere). Levels below the surface (masked in
    model output) are missing data.

    Variable ``diagnostic_kwargs``: ``hemisphere`` (``"north"`` default or
    ``"south"``), ``aggregation``, ``grid``, ``pressure`` (streamfunction level
    in Pa, default 50000) and ``search_latitudes`` (absolute latitude range of
    the extremum, default ``[0, 45]``).
    """

    _index_units = "degrees_north"
    _index_name = "hadley_cell_edge"

    def _index_values(self, cube: Cube, options: dict[str, Any]) -> np.ndarray:
        """Compute the edge latitude of every period."""
        hemisphere = str(options.get("hemisphere", "north"))
        pressure = float(options.get("pressure", 50000.0))
        search = tuple(options.get("search_latitudes", (0.0, 45.0)))
        plev_coord = cube.coord(_PRESSURE_COORD)
        plev = plev_coord.units.convert(plev_coord.points.astype(float), "Pa")
        lat = cube.coord("latitude").points.astype(float)
        va = cube_to_array(cube, "time", _PRESSURE_COORD, "latitude", "longitude")
        zonal = nan_mean(va, axis=-1)
        edges = np.full(zonal.shape[0], np.nan)
        for step in range(zonal.shape[0]):
            psi = _physics.meridional_streamfunction(zonal[step], plev, lat)
            psi_level = _physics.interpolate_to_pressure(psi, plev, pressure)
            edges[step] = _physics.hadley_cell_edge(
                psi_level,
                lat,
                hemisphere,
                search_latitudes=(float(search[0]), float(search[1])),
            )
        return edges


class EddyDrivenJetLatitude(_CirculationIndex):
    """Manage eddy-driven jet latitude diagnostics (850 hPa zonal wind maximum).

    Input: monthly ``ua`` at 850 hPa (select it with ``vertical_level`` in the
    variable definition; a cube that still has a pressure axis is interpolated
    to 850 hPa). The zonal-mean wind of each period has its maximum between 20
    and 70 degrees located by a parabolic fit
    (:func:`~climateeval.diags._physics.jet_latitude`). Set the variable's
    ``min_lon``/``max_lon`` for a basin jet.

    Variable ``diagnostic_kwargs``: ``hemisphere`` (``"north"`` default or
    ``"south"``), ``aggregation``, ``grid``, ``lat_range`` (absolute latitude
    range, default ``[20, 70]``), ``half_width`` (points each side of the
    maximum in the fit, default 2) and ``pressure`` (Pa, default 85000, only
    used if the cube has a pressure axis).
    """

    _index_units = "degrees_north"
    _index_name = "eddy_driven_jet_latitude"

    def _prepare_field(self, cube: Cube, variable: Variable) -> Cube:
        """Interpolate to the jet level if the cube still has a pressure axis."""
        if cube.coords(_PRESSURE_COORD, dim_coords=True):
            pressure = float(variable.diagnostic_kwargs.get("pressure", 85000.0))
            cube = extract_levels(
                cube,
                [pressure],
                "linear",
                coordinate=_PRESSURE_COORD,
            )
        return cube

    def _index_values(self, cube: Cube, options: dict[str, Any]) -> np.ndarray:
        """Compute the jet latitude of every period."""
        hemisphere = str(options.get("hemisphere", "north"))
        lat_range = tuple(options.get("lat_range", (20.0, 70.0)))
        half_width = int(options.get("half_width", 2))
        lat = cube.coord("latitude").points.astype(float)
        ua = cube_to_array(cube, "time", "latitude", "longitude")
        zonal = nan_mean(ua, axis=-1)
        return np.array(
            [
                _physics.jet_latitude(
                    u,
                    lat,
                    hemisphere,
                    lat_range=(float(lat_range[0]), float(lat_range[1])),
                    half_width=half_width,
                )
                for u in zonal
            ],
        )


class WalkerCirculationStrength(_CirculationIndex):
    """Manage Walker-circulation strength diagnostics (Vecchi et al., 2006).

    Input: monthly ``psl``. The index is the area-mean sea-level pressure of
    the central-east Pacific (5S-5N, 160W-80W) minus that of the Indian
    Ocean-west Pacific (5S-5N, 80E-160E)
    (:func:`~climateeval.diags._physics.walker_circulation_strength`), in the
    unit of ``psl`` (set ``units`` of the variable, e.g. ``hPa``).

    Variable ``diagnostic_kwargs``: ``aggregation`` (default ``"annual"``; use
    ``"monthly"`` for a monthly series), ``grid``, ``east_box`` and
    ``west_box`` (``[[south, north], [west, east]]``).
    """

    _index_units = "Pa"
    _index_name = "walker_circulation_strength"

    def _output_units(self, cube: Cube) -> str:
        """Get the units of the input, which the difference keeps."""
        return str(cube.units)

    def _index_values(self, cube: Cube, options: dict[str, Any]) -> np.ndarray:
        """Compute the pressure difference of every period."""
        defaults = {
            "east_box": ((-5.0, 5.0), (200.0, 280.0)),
            "west_box": ((-5.0, 5.0), (80.0, 160.0)),
        }
        east = options.get("east_box", defaults["east_box"])
        west = options.get("west_box", defaults["west_box"])

        def as_box(box: Any) -> tuple[tuple[float, float], tuple[float, float]]:
            return (
                (float(box[0][0]), float(box[0][1])),
                (float(box[1][0]), float(box[1][1])),
            )

        lat = cube.coord("latitude").points.astype(float)
        lon = cube.coord("longitude").points.astype(float)
        psl = cube_to_array(cube, "time", "latitude", "longitude")
        return np.asarray(
            _physics.walker_circulation_strength(
                psl,
                lat,
                lon,
                east_box=as_box(east),
                west_box=as_box(west),
            ),
        )
