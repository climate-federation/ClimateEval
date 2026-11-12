"""Provide helpers to move between Iris cubes and plain NumPy arrays.

The physics in :mod:`climateeval.diags._physics` takes arrays with NaN for
missing values; these helpers convert cubes (masked arrays whose mask marks
missing data, e.g. levels below the surface) accordingly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from iris.cube import Cube


def cube_to_array(cube: Cube, *coord_names: str) -> np.ndarray:
    """Get cube data as a float array in the given coordinate order.

    A masked array's mask becomes NaN (``np.asarray`` would expose the values
    under the mask, e.g. below the surface). Coordinates that the cube lacks
    as dimensions (a scalar pressure after level extraction) are skipped.

    Parameters
    ----------
    cube:
        Cube with one dimension per named coordinate.
    *coord_names:
        Names of the coordinates, in the desired axis order.

    Returns
    -------
    :
        Data with NaN where the cube is masked.

    """
    data = cube.data
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data.astype(float), np.nan)
    data = np.asarray(data, dtype=float)
    dims = [
        cube.coord_dims(name)[0]
        for name in coord_names
        if cube.coords(name) and cube.coord_dims(name)
    ]
    return np.transpose(data, dims)


def nan_mean(data: np.ndarray, axis: int) -> np.ndarray:
    """Get the mean ignoring NaN, NaN (without warning) if all are missing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(data, axis=axis)
