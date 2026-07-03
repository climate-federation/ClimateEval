"""Provide the QBO (Quasi-Biennial Oscillation) diagnostic.

The QBO is diagnosed from monthly-mean zonal-mean zonal wind (ua) averaged
over a narrow equatorial band (±5°) and the stratospheric pressure range
1–100 hPa.  Because the QBO period is ~28 months, a minimum of 5 years of
data is required; shorter simulations raise a ValueError so the diagnostic
is silently skipped.

Output cube coordinates: (time, air_pressure).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from esmvalcore.preprocessor import (
    extract_levels,
    extract_region,
    meridional_statistics,
    regrid,
    zonal_statistics,
)
from loguru import logger

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable

logger = logger.opt(colors=True)

# Stratospheric pressure levels: 100 → 1 hPa (in Pa)
_QBO_PLEV_LEVELS: list[float] = [
    10000.0,  # 100 hPa
    7000.0,  #  70 hPa
    5000.0,  #  50 hPa
    3000.0,  #  30 hPa
    2000.0,  #  20 hPa
    1000.0,  #  10 hPa
    500.0,  #   5 hPa
    100.0,  #   1 hPa
]

_LAT_BAND = 5.0  # ±degrees for equatorial average
_MIN_YEARS = 5.0  # minimum simulation length to compute QBO


class QBO(SimpleDiagnostic):
    """QBO diagnostic: stratospheric ua on a pressure × time diagram."""

    _final_coordinates = (Coordinate("time"), Coordinate("plev"))
    _MIN_YEARS: ClassVar[float] = _MIN_YEARS

    def _preprocess(self, cube: Cube, variable: Variable) -> Cube:  # noqa: ARG002
        """Preprocess to (time, air_pressure) field."""
        # Guard: need ≥5 years to see a QBO signal
        tc = cube.coord("time")
        t0, t1 = tc.units.num2date([tc.points[0], tc.points[-1]])
        n_years = (t1 - t0).days / 365.25
        if n_years < self._MIN_YEARS:
            msg = (
                f"QBO diagnostic requires ≥{self._MIN_YEARS:.0f} years of data; "
                f"this dataset spans {n_years:.1f} years — skipping."
            )
            logger.warning(msg)
            raise ValueError(msg)

        with setup_esmvaltool_config_and_logging():
            cube = extract_levels(
                cube,
                _QBO_PLEV_LEVELS,
                "linear",
                coordinate="air_pressure",
            )
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = extract_region(
                cube,
                start_longitude=0.0,
                end_longitude=360.0,
                start_latitude=-_LAT_BAND,
                end_latitude=_LAT_BAND,
            )
            cube = zonal_statistics(cube, "mean")
            cube = meridional_statistics(cube, "mean")
        return cube
