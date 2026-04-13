from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris.cube import Cube

    from benchmarking.diags import Diagnostic


def assert_preprocessing_works(diag: Diagnostic, test_cube: Cube) -> None:
    preprocessed_cube = diag._preprocess(test_cube)
    dim_coords = [c.name() for c in preprocessed_cube.coords(dim_coords=True)]
    assert set(dim_coords) == set(diag.FINAL_COORDINATES)
