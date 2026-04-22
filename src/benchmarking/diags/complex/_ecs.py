"""Provide Effective Climate Sensitivity (ECS) diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris.cube import Cube

from benchmarking import Variable
from benchmarking.diags.complex._base import ComplexDiagnostic

if TYPE_CHECKING:
    from benchmarking.data import ComplexDataSource


class ECS(ComplexDiagnostic):
    """Manage ECS diagnostics."""

    REQUIRED_DATA_KEYS = ("4xco2", "picontrol")

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        tas = Variable("tas", "tas", "mon")
        rtnt = Variable("rtnt", "rtnt", "mon")
        from iris.coords import DimCoord

        x_coord = DimCoord([1, 2], var_name="x")
        y_coord = DimCoord([1, 2, 3], var_name="y")
        return {
            tas: Cube([2.0, 3.0], dim_coords_and_dims=[(x_coord, 0)]),
            rtnt: Cube([4.0, 5.0, 6.0], dim_coords_and_dims=[(y_coord, 0)]),
        }
