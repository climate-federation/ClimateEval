"""Provide sea-ice diagnostics.

These diagnostics evaluate the total sea-ice area or extent over a hemisphere
(the region is selected per variable via ``min_lat``/``max_lat`` in the suite).
Unlike the generic area-mean diagnostics, both are obtained by an area-weighted
*sum* over the region and converted to ``1e6 km2``:

* **area** is the sum of the sea-ice concentration weighted by the cell area,
  i.e. the area actually covered by ice;
* **extent** is the sum of the area of all cells whose concentration reaches
  :data:`SEA_ICE_EXTENT_THRESHOLD` (15 %), the definition used by NSIDC and
  OSI SAF and the one most sea-ice observations are reported with.

To evaluate a single calendar month (e.g. the September minimum in the Northern
Hemisphere or the February minimum in the Southern Hemisphere), add an
``esmvalcore.preprocessor.extract_month`` step to the variable's
``additional_preprocessors`` in the suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    climate_statistics,
    convert_units,
    regrid,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import SEA_ICE_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


#: Target units of the integrated sea-ice area.
_SEA_ICE_AREA_UNITS = "1e6 km2"

#: Units the sea-ice concentration threshold is given in.
_SEA_ICE_CONCENTRATION_UNITS = "%"

#: Sea-ice concentration (in %) from which a grid cell counts as ice-covered
#: for the sea-ice *extent* (the NSIDC / OSI SAF definition).
SEA_ICE_EXTENT_THRESHOLD = 15.0


def _integrate_sea_ice(cube: Cube, concentration_threshold: float | None) -> Cube:
    """Regrid, optionally threshold, and area-integrate sea-ice concentration.

    Parameters
    ----------
    cube:
        Cube of sea-ice concentration.
    concentration_threshold:
        If given, all cells whose concentration reaches this threshold (in %)
        count fully and all others not at all, which turns the integral from a
        sea-ice *area* into a sea-ice *extent*. The threshold is applied after
        the regridding, i.e. on the common grid both the model and the
        reference are evaluated on.

    Returns
    -------
    :
        Cube of the integrated sea-ice area or extent in ``1e6 km2``.

    """
    cube = regrid(
        cube,
        SEA_ICE_GRID,
        "linear",
        cache_weights=True,
    )  # to ensure we are on a regular grid
    if concentration_threshold is not None:
        cube = convert_units(cube, _SEA_ICE_CONCENTRATION_UNITS)
        # Fully ice-covered (100 %) where the threshold is reached, ice-free
        # elsewhere; masked cells stay masked
        cube = cube.copy(data=(cube.core_data() >= concentration_threshold) * 100.0)
    cube = area_statistics(cube, "sum")
    return convert_units(cube, _SEA_ICE_AREA_UNITS)


class SeaIceAreaAnnualCycle(SimpleDiagnostic):
    """Manage sea-ice area annual cycle diagnostics.

    Computes the monthly climatology of the total sea-ice area over the region
    selected for the variable (e.g. a hemisphere), in ``1e6 km2``.
    """

    _final_coordinates = (Coordinate("month_number"),)

    #: Concentration threshold (in %) of the integral; `None` integrates the
    #: concentration itself, i.e. computes an area instead of an extent.
    _concentration_threshold: ClassVar[float | None] = None

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = _integrate_sea_ice(cube, self._concentration_threshold)
            cube = climate_statistics(cube, "mean", "month")
        return cube  # noqa: RET504


class SeaIceExtentAnnualCycle(SeaIceAreaAnnualCycle):
    """Manage sea-ice extent annual cycle diagnostics.

    Like :class:`SeaIceAreaAnnualCycle`, but sums the area of all cells whose
    sea-ice concentration reaches :data:`SEA_ICE_EXTENT_THRESHOLD` instead of
    the ice-covered area itself.
    """

    _concentration_threshold: ClassVar[float | None] = SEA_ICE_EXTENT_THRESHOLD


class SeaIceAreaTimeSeries(SimpleDiagnostic):
    """Manage sea-ice area annual mean time series diagnostics.

    Computes an annual time series of the total sea-ice area over the region
    selected for the variable, in ``1e6 km2``. Combine with an
    ``extract_month`` additional preprocessor in the suite to obtain the time
    series of a single calendar month (e.g. the seasonal sea-ice minimum).
    """

    _final_coordinates = (Coordinate("time"),)

    _annual_operator: ClassVar[str] = "mean"

    #: Concentration threshold (in %) of the integral; `None` integrates the
    #: concentration itself, i.e. computes an area instead of an extent.
    _concentration_threshold: ClassVar[float | None] = None

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = _integrate_sea_ice(cube, self._concentration_threshold)
            cube = annual_statistics(cube, self._annual_operator)
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504


class SeaIceExtentTimeSeries(SeaIceAreaTimeSeries):
    """Manage sea-ice extent annual mean time series diagnostics.

    Like :class:`SeaIceAreaTimeSeries`, but sums the area of all cells whose
    sea-ice concentration reaches :data:`SEA_ICE_EXTENT_THRESHOLD` instead of
    the ice-covered area itself.
    """

    _concentration_threshold: ClassVar[float | None] = SEA_ICE_EXTENT_THRESHOLD
