"""Load CMOR-compliant model output for benchmarking.

Handles the layouts model output typically comes in:

- a flat directory of NetCDF files (or a single file),
- a CMOR DRS tree (per-variable subdirectories with ``<grid>/<version>``
  levels, as found in ESGF/CMIP6 archives), selecting the preferred grid and
  the latest version,
- time-split archives (e.g. one file per year), which are concatenated to one
  cube per variable.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import iris
from iris.cube import CubeList
from loguru import logger

from climateeval._combine import concatenate_cubes

_VERSION_RE = re.compile(r"^(v\d+|latest)$")

#: CMOR trailing date token: ``<start>-<end>`` at any CMOR precision (``YYYY``
#: for ``yr`` up to ``YYYYMMDDHHMMSS`` for ``subhr``) with the optional
#: ``-clim`` suffix of climatology files. Anything else is not a date range.
_DATE_TOKEN_RE = re.compile(r"^(\d{4,14})-(\d{4,14})(?:-clim)?$")


def _timerange_years(timerange: str | None) -> tuple[int, int] | None:
    """Parse an ISO-like timerange (``19900101/20091231``) into years."""
    if timerange is None:
        return None
    try:
        start, end = timerange.split("/", 1)
        return int(start[:4]), int(end[:4])
    except ValueError:
        return None


def _overlaps_years(filename: str, years: tuple[int, int] | None) -> bool:
    """Check the CMOR filename date token against a year range.

    The comparison is at year precision, inclusive at both ends, so a file
    that merely straddles the window is kept. Files whose trailing token is
    not a ``<start>-<end>`` date range (``fx`` fields, observational products
    with their own naming, a version such as ``v5-1``) are included to be safe.
    """
    if years is None:
        return True
    match = _DATE_TOKEN_RE.match(Path(filename).stem.rsplit("_", 1)[-1])
    if match is None:
        return True
    start_str, end_str = match.groups()
    return int(end_str[:4]) >= years[0] and int(start_str[:4]) <= years[1]


def _select_drs_leaves(leaf_dirs: list[Path], preferred_grid: str) -> list[Path]:
    """Reduce DRS leaf directories to one per variable.

    Leaves following the ``<var>/<grid>/<version>`` pattern are grouped by
    variable directory; the preferred grid (falling back to the first grid
    alphabetically) and the latest version win. Leaves not following the
    pattern are kept as-is.
    """
    plain: list[Path] = []
    by_var: dict[Path, dict[str, list[Path]]] = {}
    for leaf in leaf_dirs:
        if _VERSION_RE.match(leaf.name) and leaf.parent != leaf:
            grid_dir = leaf.parent
            by_var.setdefault(grid_dir.parent, {}).setdefault(
                grid_dir.name,
                [],
            ).append(leaf)
        else:
            plain.append(leaf)

    selected = plain
    for var_dir, grids in by_var.items():
        grid = preferred_grid if preferred_grid in grids else sorted(grids)[0]
        version_leaf = sorted(grids[grid])[-1]
        if len(grids) > 1 or len(grids[grid]) > 1:
            logger.debug(
                f"{var_dir}: selected grid '{grid}', version '{version_leaf.name}'",
            )
        selected.append(version_leaf)
    return sorted(selected)


def load_cmor_dir(
    path: Path | str,
    *,
    timerange: str | None = None,
    preferred_grid: str = "gn",
) -> CubeList:
    """Load CMOR-compliant NetCDF output from a file, directory, or DRS tree.

    Parameters
    ----------
    path:
        A NetCDF file, a flat directory of NetCDF files, or the root of a
        CMOR DRS tree (e.g. an ``Amon`` table directory with per-variable
        subdirectories). For DRS trees, the preferred grid and the latest
        version are selected per variable.
    timerange:
        Optional ISO-like timerange (``19900101/20091231``). Files whose
        filename date token does not overlap it are skipped.
    preferred_grid:
        Grid label preferred when a DRS variable is available on several
        grids (default: the native grid ``gn``).

    Returns
    -------
    :
        One cube per variable; time-split files are concatenated.

    """
    path = Path(path)
    years = _timerange_years(timerange)

    groups: list[list[Path]]
    if path.is_file():
        groups = [[path]]
    else:
        flat = sorted(path.glob("*.nc"))
        if flat:
            groups = [flat]
        else:
            leaf_dirs = sorted({f.parent for f in path.rglob("*.nc")})
            groups = [
                sorted(leaf.glob("*.nc"))
                for leaf in _select_drs_leaves(leaf_dirs, preferred_grid)
            ]

    cubes: CubeList = CubeList([])
    for group in groups:
        relevant = [f for f in group if _overlaps_years(f.name, years)]
        if not relevant:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = iris.load([str(f) for f in relevant])
            # Coordinate metadata differs between the files of one CMIP6 leaf
            # often enough that concatenation silently returns one cube per
            # file; `concatenate_cubes` reconciles what it safely can and warns
            # about whatever is left (see climateeval._combine).
            loaded = concatenate_cubes(loaded, source=str(relevant[0].parent))
        cubes.extend(loaded)

    logger.info(f"Loaded {len(cubes)} cube(s) from {path}")
    return cubes
