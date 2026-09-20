"""Normalise coordinate metadata so that time-split files concatenate.

``iris`` concatenates two cubes only when everything about them except the
concatenation axis matches exactly. :func:`iris.util.equalise_attributes` and
:func:`iris.util.unify_time_units` reconcile the *cube's* attributes and the
time epoch, but nothing reconciles the **coordinates'** own metadata, and real
CMIP6 archives disagree there constantly:

===============================  =========================================
disagreement                     example in the DKRZ CMIP6 replica
===============================  =========================================
per-file coordinate attribute    IPSL-CM6A-LR writes ``time:time_origin``,
                                 a different value in every file
coordinate ``long_name`` case    AWI-CM-1-1-MR: ``lat:long_name`` is
                                 ``latitude`` in the 1850 file and
                                 ``Latitude`` in every later one
coordinate ``long_name`` wording EC-Earth3-Veg-LR: the ``i``/``j`` index
                                 coordinates are described two ways
calendar spelling                CESM2 alternates ``noleap`` and
                                 ``365_day``; EC-Earth3-Veg alternates
                                 ``gregorian`` and ``proleptic_gregorian``
float32 vs float64 coordinates   MIROC6 publishes the same ocean grid at
                                 two precisions (~2e-6 degrees apart)
===============================  =========================================

When that happens ``concatenate()`` does not fail — it silently returns
**several cubes for one variable**, and every downstream diagnostic then sees a
truncated record (or picks one segment at random). That is the failure mode
this module exists to remove: it normalises what can safely be normalised, and
:func:`concatenate_cubes` **warns, naming the variable and the reason**, when a
variable still comes back in pieces.

Nothing here changes data values. The two judgement calls are documented where
they are made: coordinate values are only ever snapped together when they agree
to within :data:`COORD_RTOL` (float32 round-trip error), and calendars are only
ever unified inside the Gregorian family, for records that lie entirely after
the 1582 Julian/Gregorian transition, where those calendars coincide exactly.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from cf_units import Unit
from iris.cube import CubeList
from iris.exceptions import ConcatenateError
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from iris.coords import Coord
    from iris.cube import Cube

logger = logger.opt(colors=True)

#: Calendars that are the *same* calendar for any date after the Julian ->
#: Gregorian transition. ``cf_units`` already treats ``gregorian`` as a synonym
#: of ``standard``; ``proleptic_gregorian`` differs from them only before 1582.
GREGORIAN_FAMILY = frozenset({"standard", "gregorian", "proleptic_gregorian"})

#: Year of the Julian -> Gregorian transition. Calendars in
#: :data:`GREGORIAN_FAMILY` are unified only for records that start after it.
GREGORIAN_TRANSITION_YEAR = 1582

#: Two of anything: the smallest number that can disagree.
_A_PAIR = 2

#: Relative tolerance within which two coordinates count as the same
#: coordinate written at two precisions. A float32 round trip of a latitude in
#: degrees is wrong by ~1e-8 relative (~1e-6 degrees, a few centimetres on the
#: ground); anything further apart is a different grid and is left alone.
COORD_RTOL = 1e-6


def _coord_key(coord: Coord) -> str:
    """Match coordinates across cubes even when their names disagree.

    ``coord.name()`` falls back to ``long_name``, which is exactly what
    differs in the EC-Earth3-Veg-LR ``i``/``j`` case, so the NetCDF variable
    name is tried first.
    """
    return coord.var_name or coord.name()


def _group_by_variable(cubes: Iterable[Cube]) -> dict[str, list[Cube]]:
    """Cubes grouped by the variable they hold."""
    groups: dict[str, list[Cube]] = {}
    for cube in cubes:
        groups.setdefault(cube.var_name or cube.name(), []).append(cube)
    return groups


def _drop_differing_attributes(coords: Sequence[Coord]) -> None:
    """Remove coordinate attributes that are not identical in every cube.

    A per-file provenance attribute such as IPSL-CM6A-LR's ``time_origin`` is
    metadata about the file, not about the coordinate, and keeping it splits
    the record into one cube per file.
    """
    common: dict[str, Any] = dict(coords[0].attributes)
    for coord in coords[1:]:
        for key in list(common):
            if key not in coord.attributes or not _equal(
                common[key],
                coord.attributes[key],
            ):
                del common[key]
    for coord in coords:
        if dict(coord.attributes) != common:
            coord.attributes = dict(common)


def _equal(left: Any, right: Any) -> bool:
    """Value equality that also copes with arrays."""
    try:
        return bool(np.all(left == right))
    except Exception:  # noqa: BLE001
        return False


def _unify_names(coords: Sequence[Coord]) -> None:
    """Give matched coordinates one ``long_name``/``var_name``.

    Only when they already agree on ``standard_name`` — that is what says they
    are the same physical quantity. A coordinate with no ``standard_name``
    (an unstructured grid's ``i``/``j`` index) qualifies through the matching
    ``var_name`` that brought it here.
    """
    standard_names = {coord.standard_name for coord in coords}
    if len(standard_names) > 1:
        return
    reference = coords[0]
    for coord in coords[1:]:
        coord.long_name = reference.long_name
        coord.var_name = reference.var_name


def _upcast(coords: Sequence[Coord]) -> list[bool]:
    """Cast float32 coordinate points and bounds to float64 (lossless).

    Points and bounds are handled independently: CESM2 publishes ``latitude``
    with float64 points but float32 *bounds* in its historical files and
    float64 bounds in its SSP files, which is enough to split the record.

    Returns, per coordinate, whether it was already fully float64 — the higher
    precision source is the better reference for :func:`_align_values`.
    """
    was_double: list[bool] = []
    for coord in coords:
        double = coord.points.dtype == np.float64
        if coord.points.dtype == np.float32:
            coord.points = coord.points.astype(np.float64)
        if coord.has_bounds():
            double = double and coord.bounds.dtype == np.float64
            if coord.bounds.dtype == np.float32:
                coord.bounds = coord.bounds.astype(np.float64)
        was_double.append(bool(double))
    return was_double


def _close(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether two coordinate arrays are the same values at two precisions."""
    return (
        left.shape == right.shape
        and not np.array_equal(left, right)
        and bool(np.allclose(left, right, rtol=COORD_RTOL, atol=0.0))
    )


def _align_values(coords: Sequence[Coord], was_double: Sequence[bool]) -> None:
    """Snap coordinates that agree to within :data:`COORD_RTOL` together.

    This is the float32/float64 case only: the same grid written at two
    precisions, differing by the float32 round-trip error. Points and bounds
    are snapped independently, because CMIP6 files disagree on them
    independently. Coordinates that differ by more than that are a genuinely
    different grid (KIOST-ESM publishes ``tos`` on two of them) or corrupt
    (NorESM2-LM writes a 1e30 fill into a latitude), and are left exactly as
    they are so that the concatenation fails loudly instead of quietly
    stitching two grids together.
    """
    reference = coords[was_double.index(True)] if any(was_double) else coords[0]
    for coord in coords:
        if coord is reference:
            continue
        if _close(coord.points, reference.points):
            coord.points = reference.points.copy()
        if (
            coord.has_bounds()
            and reference.has_bounds()
            and _close(coord.bounds, reference.bounds)
        ):
            coord.bounds = reference.bounds.copy()


def _unify_calendars(coords: Sequence[Coord]) -> None:
    """Give time coordinates one calendar, where that is exactly true.

    Only inside :data:`GREGORIAN_FAMILY`, and only when every record starts
    after :data:`GREGORIAN_TRANSITION_YEAR`, where those calendars agree date
    for date — so the point values keep their meaning and are not touched. A
    ``noleap``/``360_day`` record is never converted: ``cf_units`` already
    treats ``noleap`` and ``365_day`` as one calendar, and any other mismatch
    is a real difference that must surface.
    """
    units = [coord.units for coord in coords]
    if not all(unit.is_time_reference() for unit in units):
        return
    calendars = {unit.calendar for unit in units}
    if len(calendars) < _A_PAIR or not calendars <= GREGORIAN_FAMILY:
        return
    for coord, unit in zip(coords, units, strict=True):
        if not coord.points.size:
            continue
        if unit.num2date(coord.points.min()).year <= GREGORIAN_TRANSITION_YEAR:
            return
    for coord in coords:
        coord.units = Unit(coord.units.origin, calendar="standard")


def _unify_cube_names(cubes: Sequence[Cube]) -> None:
    """Give cubes of one variable one set of names.

    IPSL-CM6A-LR labels ``prw`` ``atmosphere_water_vapor_content`` in some
    files of a run and ``atmosphere_mass_content_of_water_vapor`` in others.
    The cubes are already known to be the same variable (they share a
    ``var_name``), so the majority spelling wins; a tie keeps the first.
    """
    if len({cube.var_name for cube in cubes}) != 1 or cubes[0].var_name is None:
        return
    for attribute in ("standard_name", "long_name", "units"):
        values = [getattr(cube, attribute) for cube in cubes]
        if len({str(value) for value in values}) < _A_PAIR:
            continue
        winner = Counter(str(value) for value in values).most_common(1)[0][0]
        chosen = next(v for v in values if str(v) == winner)
        for cube in cubes:
            with contextlib.suppress(ValueError):
                setattr(cube, attribute, chosen)


def normalise_coordinates(cubes: CubeList) -> CubeList:
    """Reconcile the coordinate metadata of cubes holding the same variable.

    Modifies ``cubes`` in place and returns it. Every step is metadata-only
    except the float32 -> float64 upcast, which is lossless, and the snapping
    of coordinates that already agree to within :data:`COORD_RTOL`.
    """
    for group in _group_by_variable(cubes).values():
        if len(group) < _A_PAIR:
            continue
        _unify_cube_names(group)
        keyed = [
            {_coord_key(coord): coord for coord in cube.coords()} for cube in group
        ]
        shared = set(keyed[0])
        for mapping in keyed[1:]:
            shared &= set(mapping)
        for key in sorted(shared):
            coords = [mapping[key] for mapping in keyed]
            _drop_differing_attributes(coords)
            _unify_names(coords)
            was_double = _upcast(coords)
            if not coords[0].units.is_time_reference():
                _align_values(coords, was_double)
            _unify_calendars(coords)
    return cubes


def _describe_split(cubes: Sequence[Cube]) -> str:
    """Why iris refused to concatenate these cubes, in one line."""
    try:
        CubeList(list(cubes)).concatenate_cube()
    except ConcatenateError as exc:
        return "; ".join(str(exc).split("\n")).strip()
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return "unknown (they concatenate in isolation)"


def warn_on_split_variables(cubes: Iterable[Cube], source: str = "") -> None:
    """Log a warning for every variable that came back as several cubes.

    A split variable is the silent failure this module is about: the caller
    gets a short record instead of an error, so say so explicitly, naming the
    variable, its segments and iris' own reason.
    """
    where = f" in {source}" if source else ""
    for var_name, group in _group_by_variable(cubes).items():
        if len(group) < _A_PAIR:
            continue
        if not any(cube.coords("time") for cube in group):
            # Not the failure this is about. Two cubes of one name with no
            # time axis are two ancillary fields that happen to share a name
            # (a CMORized observation file carries `area` on two grids), not a
            # record split in half.
            continue
        spans = []
        for cube in group:
            if cube.coords("time"):
                coord = cube.coord("time")
                first = coord.units.num2date(coord.points[0])
                last = coord.units.num2date(coord.points[-1])
                spans.append(f"{first:%Y-%m}..{last:%Y-%m}")
            else:
                spans.append("no time axis")
        logger.warning(
            f"Variable '{var_name}'{where} did not concatenate into a single "
            f"cube: {len(group)} segments ({', '.join(spans)}). Diagnostics "
            f"will see only part of the record. Reason: "
            f"{_describe_split(group)}",
        )


def concatenate_cubes(cubes: CubeList, source: str = "") -> CubeList:
    """Normalise coordinate metadata, concatenate, and report what did not."""
    from iris.util import equalise_attributes, unify_time_units  # noqa: PLC0415

    equalise_attributes(cubes)
    unify_time_units(cubes)
    normalise_coordinates(cubes)
    result = cubes.concatenate()
    warn_on_split_variables(result, source)
    return result
