"""Provide access to the CMIP6 branch time of a child experiment.

A CMIP6 experiment that branches off a control run records where it did so
in the global attributes ``branch_time_in_parent`` and
``parent_time_units``. Anomalies of the child against the *parallel* segment
of the control - the control years running alongside the child's - remove
the control's own drift, which anomalies against the control's long-term
mean do not (Schlund et al. 2020, https://doi.org/10.5194/esd-11-1233-2020).

"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from cf_units import Unit
from loguru import logger

if TYPE_CHECKING:
    from iris.cube import CubeList
    from xarray import Dataset

logger = logger.opt(colors=True)

_DASHED_CLOCK_TIME_UNITS = re.compile(
    r"^(?P<unit>\w+ since \d{1,4}-\d{1,2}-\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})$",
)
"""Time units spelled ``days since 1850-01-01-00-00-00``.

The MOHC models (HadGEM3-GC31, UKESM1) write ``parent_time_units`` with the
clock time joined on by dashes, which ``cf_units`` cannot parse (ESMValCore
carries the same fix for these models).
"""


def normalise_time_units(units: str) -> str:
    """Get a time-unit string in a spelling ``cf_units`` can parse.

    Parameters
    ----------
    units:
        Time units as found in the global attributes.

    Returns
    -------
    :
        The units with a dashed clock time (``...-00-00-00``) rewritten as
        ``... 00:00:00``; any other string unchanged.

    """
    match = _DASHED_CLOCK_TIME_UNITS.match(units.strip())
    if match is None:
        return units
    hour, minute, second = match.group(2, 3, 4)
    return f"{match.group('unit')} {hour}:{minute}:{second}"


def global_attribute(data: CubeList | Dataset, key: str) -> Any:
    """Get a global attribute of an experiment's data, if it survived loading.

    ``load_cmor_dir`` equalises the attributes of the time-split files of one
    variable, which drops the attributes that differ between them
    (``creation_date``, ``tracking_id``) but keeps the run-level ones; the
    branch attributes are identical across a run's files.

    Parameters
    ----------
    data:
        Cubes or dataset of one experiment.
    key:
        Name of the global attribute.

    Returns
    -------
    :
        The attribute's value, or ``None`` if no cube carries it.

    """
    attributes = getattr(data, "attrs", None)  # an xarray Dataset
    if attributes is not None and key in attributes:
        return attributes[key]
    try:
        cubes = list(data)
    except TypeError:
        return None
    for cube in cubes:
        value = getattr(cube, "attributes", {}).get(key)
        if value is not None:
            return value
    return None


def calendar(data: CubeList | Dataset) -> str | None:
    """Get the calendar of an experiment's time axis.

    Parameters
    ----------
    data:
        Cubes or dataset of one experiment.

    Returns
    -------
    :
        Calendar name, or ``None`` if no cube has a time coordinate. The
        parent run shares the calendar of its child.

    """
    try:
        cubes = list(data)
    except TypeError:
        return None
    for cube in cubes:
        try:
            return str(cube.coord("time").units.calendar)
        except Exception:  # noqa: BLE001, S112 - not every cube has a time axis
            continue
    return None


def branch_year_in_parent(
    data: CubeList | Dataset,
    *,
    context: str = "",
) -> int | None:
    """Get the calendar year of the parent run a child experiment branched at.

    Decodes ``branch_time_in_parent`` with ``parent_time_units`` on the
    child's own calendar.

    Parameters
    ----------
    data:
        Cubes or dataset of the child experiment.
    context:
        Name of the calling diagnostic, used in the warnings.

    Returns
    -------
    :
        Calendar year of the parent run, or ``None`` if the metadata is
        missing or cannot be decoded (a warning says which).

    """
    prefix = f"Diagnostic '{context}': " if context else ""
    branch_time = global_attribute(data, "branch_time_in_parent")
    parent_time_units = global_attribute(data, "parent_time_units")
    if branch_time is None or parent_time_units is None:
        logger.warning(
            f"{prefix}the experiment carries no branch_time_in_parent / "
            f"parent_time_units attributes",
        )
        return None
    child_calendar = calendar(data)
    time_units = normalise_time_units(str(parent_time_units))
    try:
        unit = (
            Unit(time_units, calendar=child_calendar)
            if child_calendar
            else Unit(time_units)
        )
        return int(unit.num2date(float(branch_time)).year)
    except Exception as exc:  # noqa: BLE001 - malformed metadata is common
        logger.warning(
            f"{prefix}cannot decode branch_time_in_parent={branch_time!r} in "
            f"'{parent_time_units}': {exc}",
        )
        return None


def parallel_control_window(
    window: tuple[int, int],
    *,
    branch_year: int,
    child_first_year: int,
) -> tuple[int, int]:
    """Get the control years running parallel to a window of a child run.

    Year ``Y`` of the child corresponds to year
    ``Y - child_first_year + branch_year`` of its parent.

    Parameters
    ----------
    window:
        Inclusive ``(first, last)`` calendar years of the child run.
    branch_year:
        Calendar year of the parent the child branched at, from
        :func:`branch_year_in_parent`.
    child_first_year:
        First calendar year of the child run.

    Returns
    -------
    :
        Inclusive ``(first, last)`` calendar years of the parent.

    """
    offset = int(branch_year) - int(child_first_year)
    return (int(window[0]) + offset, int(window[1]) + offset)
