"""Validate model input data against suite requirements.

Backs the ``climateeval check`` subcommand: replays the code paths that
``climateeval benchmark`` runs on model input (load, variable selection by
``var_name``, coordinate checking, unit conversion, timerange coverage) on
metadata only — no preprocessing, no reference data, no realization of the
data arrays — and reports per-variable, actionable findings.

See ``docs/input-data-spec.md`` for the input requirements this checks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from esmvalcore.preprocessor import convert_units
from esmvalcore.preprocessor._derive import get_required
from iris import NameConstraint
from loguru import logger

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._loader import load_cmor_dir
from climateeval._utils import check_coordinates
from climateeval._variable import VARIABLES, Variable, VerticalLevel
from climateeval.exceptions import CoordinateError
from climateeval.suites import resolve_suite_definition_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube, CubeList

logger = logger.opt(colors=True)

Status = Literal["pass", "warn", "fail"]

_STATUS_ORDER: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True)
class CheckResult:
    """Result of a single check on a variable."""

    name: str  # found | coordinates | units | timerange | levels | sources
    status: Status
    message: str = ""


@dataclass(frozen=True)
class VariableReport:
    """All check results for one required variable."""

    var_name: str
    long_name: str
    derived: bool
    checks: tuple[CheckResult, ...]

    @property
    def status(self) -> Status:
        """Get the worst status among the checks."""
        return max((c.status for c in self.checks), key=_STATUS_ORDER.__getitem__)


@dataclass(frozen=True)
class CheckReport:
    """Full check report for a model data path."""

    model_path: str
    suites: tuple[str, ...]
    timerange: str | None
    n_cubes: int
    variables: tuple[VariableReport, ...]
    extra_var_names: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether every required variable is usable (no failures)."""
        return all(v.status != "fail" for v in self.variables)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        data = dataclasses.asdict(self)
        for entry, report in zip(data["variables"], self.variables, strict=True):
            entry["status"] = report.status
        data["passed"] = self.passed
        return data


@dataclass(frozen=True)
class VariableRequirement:
    """Input requirements for one variable, aggregated over suites."""

    var_name: str
    frequency: str
    units_targets: tuple[str, ...]
    levels: tuple[VerticalLevel, ...]


def suite_variable_requirements(
    suites: Sequence[str | Path],
) -> dict[str, VariableRequirement]:
    """Collect the input variables required by the given suites.

    Suite entries whose variable has no coordinates in ``variables.yml``
    (diagnostic outputs like ``ecs``) are skipped — they are computed, not
    read from the model input.
    """
    requirements: dict[str, VariableRequirement] = {}
    for suite in suites:
        suite_file = resolve_suite_definition_file(suite)
        spec = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
        for diagnostic in spec or []:
            for entry in diagnostic.get("variables", []):
                var_name = entry.get("var_name")
                if var_name is None:
                    continue
                var_info = VARIABLES.get(var_name)
                if var_info is None or not var_info.get("coordinates"):
                    logger.debug(
                        f"Skipping non-input variable '{var_name}' from suite "
                        f"'{suite_file.stem}'",
                    )
                    continue
                requirements[var_name] = _merge_requirement(
                    requirements.get(var_name),
                    entry,
                    str(var_info["units"]),
                )
    return requirements


def check_model_data(
    model_path: str | Path,
    *,
    suites: Sequence[str | Path],
    timerange: str | None = None,
) -> CheckReport:
    """Check model data against the input requirements of the given suites.

    Only metadata is inspected; the data arrays are never realized.
    """
    cubes = load_cmor_dir(model_path, timerange=timerange)
    requirements = suite_variable_requirements(suites)

    used_var_names: set[str] = set()
    reports = tuple(
        _check_variable(cubes, requirements[var_name], timerange, used_var_names)
        for var_name in sorted(requirements)
    )
    present = {cube.var_name for cube in cubes if cube.var_name}
    return CheckReport(
        model_path=str(model_path),
        suites=tuple(Path(str(s)).stem for s in suites),
        timerange=timerange,
        n_cubes=len(cubes),
        variables=reports,
        extra_var_names=tuple(sorted(present - used_var_names)),
    )


def _merge_requirement(
    existing: VariableRequirement | None,
    entry: dict[str, Any],
    default_units: str,
) -> VariableRequirement:
    """Merge one suite variable entry into an existing requirement."""
    # A `units` override only applies directly to the input data if no
    # additional preprocessors run before the conversion (e.g. relative
    # anomalies in Tier1_consistency_checks turn any input into "%").
    units = str(entry.get("units", "") or default_units)
    if entry.get("additional_preprocessors"):
        units = default_units

    levels: tuple[VerticalLevel, ...] = ()
    if "vertical_level" in entry:
        levels = (VerticalLevel(**entry["vertical_level"]),)

    if existing is None:
        return VariableRequirement(
            var_name=str(entry["var_name"]),
            frequency=str(entry.get("frequency", "mon")),
            units_targets=(units,),
            levels=levels,
        )
    return VariableRequirement(
        var_name=existing.var_name,
        frequency=existing.frequency,
        units_targets=tuple(dict.fromkeys((*existing.units_targets, units))),
        levels=tuple(sorted(set(existing.levels) | set(levels))),
    )


def _check_variable(
    cubes: CubeList,
    requirement: VariableRequirement,
    timerange: str | None,
    used_var_names: set[str],
) -> VariableReport:
    """Run all checks for one required variable."""
    variable = Variable(
        requirement.var_name,
        requirement.var_name,
        requirement.frequency,
    )
    checks: list[CheckResult] = []
    matches = cubes.extract(NameConstraint(var_name=variable.var_name))

    if len(matches) == 1:
        used_var_names.add(variable.var_name)
        checks.append(CheckResult("found", "pass"))
        checks.extend(
            _check_cube(matches[0], variable, requirement, timerange),
        )
    elif len(matches) > 1:
        used_var_names.add(variable.var_name)
        checks.append(
            CheckResult(
                "found",
                "fail",
                f"found {len(matches)} cubes with var_name "
                f"'{variable.var_name}' — time-split files failed to "
                f"concatenate (check for consistent calendars, units and "
                f"attributes across files)",
            ),
        )
    elif variable.derived:
        checks.extend(
            _check_derived_sources(cubes, variable, timerange, used_var_names),
        )
    else:
        checks.append(_missing_cube_result(cubes, variable.var_name))

    return VariableReport(
        var_name=variable.var_name,
        long_name=variable.long_name,
        derived=variable.derived,
        checks=tuple(checks),
    )


def _missing_cube_result(cubes: CubeList, var_name: str) -> CheckResult:
    """Build the failure result for a variable with no matching cube."""
    present = sorted({cube.var_name for cube in cubes if cube.var_name})
    present_str = ", ".join(present) if present else "none"
    return CheckResult(
        "found",
        "fail",
        f"no cube with var_name '{var_name}' (present: {present_str}) — "
        f"rename the NetCDF variable to '{var_name}'",
    )


def _check_cube(
    cube: Cube,
    variable: Variable,
    requirement: VariableRequirement,
    timerange: str | None,
    source_prefix: str = "",
) -> list[CheckResult]:
    """Check one cube against one variable's requirements."""
    checks = [_check_units(cube, requirement.units_targets, source_prefix)]

    try:
        checked = check_coordinates(cube, variable.coordinates)
    except CoordinateError as exc:
        checks.append(
            CheckResult(
                "coordinates",
                "fail",
                f"{source_prefix}{exc} — coordinates are matched by CF "
                f"standard_name, long_name or var_name; set the standard_name "
                f"on the coordinate variable",
            ),
        )
        return checks
    checks.append(CheckResult("coordinates", "pass"))

    timerange_check = _check_timerange(checked, timerange, source_prefix)
    if timerange_check is not None:
        checks.append(timerange_check)

    levels_check = _check_levels(checked, requirement.levels, source_prefix)
    if levels_check is not None:
        checks.append(levels_check)

    return checks


def _check_units(
    cube: Cube,
    units_targets: tuple[str, ...],
    source_prefix: str = "",
) -> CheckResult:
    """Check that the cube's units convert to every required target."""
    if cube.units.is_unknown():
        targets = "', '".join(units_targets)
        return CheckResult(
            "units",
            "fail",
            f"{source_prefix}units missing or invalid — set the NetCDF "
            f"'units' attribute to a unit convertible to '{targets}'",
        )
    for target in units_targets:
        try:
            # Replays the runtime conversion (including ESMValCore's special
            # standard_name-based conversions like precipitation flux <->
            # rate) on a copy; lazy data stays lazy.
            convert_units(cube.copy(), target)
        except ValueError:
            return CheckResult(
                "units",
                "fail",
                f"{source_prefix}units '{cube.units}' are not convertible to "
                f"'{target}' (for special conversions like kg m-2 s-1 <-> "
                f"mm day-1 the cube also needs the matching CF standard_name)",
            )
    conversions = [t for t in units_targets if cube.units != t]
    message = (
        f"{source_prefix}'{cube.units}' will be converted to '{', '.join(conversions)}'"
        if conversions
        else ""
    )
    return CheckResult("units", "pass", message)


def _check_timerange(
    cube: Cube,
    timerange: str | None,
    source_prefix: str = "",
) -> CheckResult | None:
    """Check that the data covers the requested timerange."""
    if timerange is None or not cube.coords("time"):
        return None
    time_coord = cube.coord("time")
    datetimes = time_coord.units.num2date(time_coord.points)
    actual_min_year = datetimes[0].year
    actual_max_year = datetimes[-1].year
    expected_min_year = int(timerange.split("/")[0][:4])
    expected_max_year = int(timerange.split("/")[1][:4])
    if expected_min_year < actual_min_year or expected_max_year > actual_max_year:
        return CheckResult(
            "timerange",
            "fail",
            f"{source_prefix}data covers {actual_min_year}-{actual_max_year} "
            f"but timerange {timerange} was requested — pass --timerange "
            f"{actual_min_year}0101/{actual_max_year}1231 or extend the data",
        )
    # For derived-variable sources the per-source coverage adds only noise
    message = (
        "" if source_prefix else f"data covers {actual_min_year}-{actual_max_year}"
    )
    return CheckResult("timerange", "pass", message)


def _check_levels(
    cube: Cube,
    levels: tuple[VerticalLevel, ...],
    source_prefix: str = "",
) -> CheckResult | None:
    """Warn about requested vertical levels outside the data's range."""
    problems: list[str] = []
    for level in levels:
        if level.extrapolate or not cube.coords(level.coordinate_name):
            continue
        coord = cube.coord(level.coordinate_name).copy()
        try:
            coord.convert_units(level.units)
        except ValueError:
            continue  # unit problems are reported by check_coordinates/units
        low = coord.points.min()
        high = coord.points.max()
        if not low <= level.value <= high:
            problems.append(
                f"level {level} outside data range {low:g}-{high:g} {level.units}",
            )
    if not levels:
        return None
    if problems:
        return CheckResult(
            "levels",
            "warn",
            f"{source_prefix}{'; '.join(problems)} — extraction will yield "
            f"masked values",
        )
    return CheckResult("levels", "pass")


def _check_derived_sources(
    cubes: CubeList,
    variable: Variable,
    timerange: str | None,
    used_var_names: set[str],
) -> list[CheckResult]:
    """Check the source variables needed to derive a variable."""
    try:
        with setup_esmvaltool_config_and_logging():
            facets = get_required(variable.var_name, "CMIP6")
    except NotImplementedError as exc:
        # e.g. phcint: no ESMValCore derivation script available
        return [
            CheckResult(
                "sources",
                "fail",
                f"{exc} — provide a pre-computed '{variable.var_name}' directly",
            ),
        ]
    optional_var_names = {
        facet["short_name"] for facet in facets if facet.get("optional", False)
    }
    required_variables = variable.get_required_variables()

    checks: list[CheckResult] = []
    present: list[str] = []
    missing_required: list[str] = []
    for source in required_variables:
        source_matches = cubes.extract(NameConstraint(var_name=source.var_name))
        if not source_matches:
            if source.var_name not in optional_var_names:
                missing_required.append(source.var_name)
            continue
        present.append(source.var_name)
        used_var_names.add(source.var_name)
        source_requirement = VariableRequirement(
            var_name=source.var_name,
            frequency=source.frequency,
            units_targets=(str(VARIABLES[source.var_name]["units"]),),
            levels=(),
        )
        checks.extend(
            _check_cube(
                source_matches[0],
                source,
                source_requirement,
                timerange,
                source_prefix=f"source '{source.var_name}': ",
            ),
        )

    all_source_names = [v.var_name for v in required_variables]
    if missing_required or not present:
        missing = missing_required or all_source_names
        checks.insert(
            0,
            CheckResult(
                "sources",
                "fail",
                f"derived from {' + '.join(all_source_names)}; missing: "
                f"{', '.join(missing)} — provide the source variable(s) or a "
                f"pre-computed '{variable.var_name}'",
            ),
        )
    else:
        checks.insert(
            0,
            CheckResult(
                "sources",
                "pass",
                f"will be derived from {', '.join(present)}",
            ),
        )
    return checks
