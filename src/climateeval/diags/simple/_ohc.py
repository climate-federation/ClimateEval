"""Provide ocean heat content diagnostics.

Note
----
The ``phcint`` (integrated ocean heat content) variable is derived by
ESMValCore. The derivation script is only available on the ESMValCore
``expand_icon_xpp_ocean`` branch, so the installed ESMValCore must include it
(``esmvalcore/preprocessor/_derive/phcint.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from esmvalcore.preprocessor import (
    annual_statistics,
    anomalies,
    area_statistics,
    axis_statistics,
    convert_units,
    regrid,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class OceanHeatContentTimeSeries(SimpleDiagnostic):
    """Manage global ocean heat content anomaly time series diagnostics.

    Computes the annual-mean global ocean heat content anomaly (relative to a
    fixed reference period) from the derived per-layer heat content ``phcint``:
    the column heat content is obtained by summing over depth (``phcint`` is
    already per unit area, so no depth weights are applied), then integrating
    over area, and finally taking annual means and anomalies in ``ZJ``.

    To restrict the diagnostic to an upper-ocean layer (e.g. 0-300 m or
    0-2000 m), add an ``esmvalcore.preprocessor.extract_volume`` step to the
    variable's ``additional_preprocessors`` in the suite.
    """

    _final_coordinates = (Coordinate("time"),)

    _units: ClassVar[str] = "ZJ"

    #: Reference period for the anomaly (inclusive), as accepted by
    #: ``esmvalcore.preprocessor.anomalies``.
    _anomaly_reference: ClassVar[dict[str, int]] = {
        "start_year": 1979,
        "start_month": 1,
        "start_day": 1,
        "end_year": 1988,
        "end_month": 12,
        "end_day": 31,
    }

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            # phcint is already per unit area, so sum over depth without weights.
            cube = axis_statistics(cube, axis="Z", operator="sum", weights=False)
            cube = area_statistics(cube, "sum")
            cube = annual_statistics(cube, "mean")
            cube = anomalies(cube, period="full", reference=self._anomaly_reference)
            cube = convert_units(cube, self._units)
            # Standardize the time coordinate so model and reference align when
            # computing metrics.
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504
