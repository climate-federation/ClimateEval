"""Provide precipitation frequency and intensity statistics from daily rain.

For each region (the variable's latitude/longitude bounds and land-sea mask),
season and year the diagnostic pools all days and grid cells (area-weighted by
cos latitude, NaN and masked cells ignored) and computes
:func:`climateeval.diags._physics.precipitation_statistics`:

``mean``
    mean precipitation (mm day-1);
``wet_fraction``, ``sdii``
    fraction of days with at least ``threshold`` (1 mm day-1) and the mean
    precipitation on those days;
``p10``, ``p90`` (``percentiles``)
    percentiles of the wet-day precipitation;
``dry_fraction``
    fraction of days below the lowest bin edge (0.03 mm day-1);
``amount_peak_rate``, ``amount_peak_height``
    rain rate and height of the maximum of the rain-amount distribution on
    logarithmic bins (Pendergrass and Hartmann, 2014);
``frequency_peak_rate``, ``frequency_peak_height``
    the same for the frequency distribution.

The regression of any of these on an index series is
:func:`climateeval.diags._physics.regress_on_index`.

Output layout. One variable of the suite (``id`` e.g. ``pr_tropics``) writes
one table with the columns ``time`` and, for every season and statistic, a
column ``<id>_<statistic>_<season>`` (for example ``pr_tropics_wet_fraction_djf``):
with ``aggregation="year"`` one row per year (the series is the annual
sequence of the statistic of that season; ``time`` is that year), with
``aggregation="period"`` a single row (the statistic of all days of the
period pooled). Seasons that are incomplete in a year are missing (NaN).

Options per variable (``diagnostic_kwargs``):

``seasons``
    list from ``ann``, ``djf``, ``mam``, ``jja``, ``son`` (default
    ``[ann, djf, jja]``); DJF of year Y is December of Y-1 with January and
    February of Y.
``statistics``
    subset of the statistics (default: all).
``aggregation``
    ``year`` (default) or ``period``.
``threshold``
    wet-day threshold in mm day-1 (default 1.0).
``percentiles``
    wet-day percentiles (default ``[10, 90]``).
``r_min``, ``ratio``, ``n_bins``
    logarithmic bins (default 0.03, 1.07, 150).
``grid``, ``regrid_scheme``
    target grid and regridding scheme (default ``1x1`` and ``area_weighted``,
    which is conservative).

No metrics against the reference are computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import ibis
import numpy as np
import pandas as pd
from esmvalcore.preprocessor import regrid, regrid_time
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags import _physics
from climateeval.diags._cube_arrays import cube_to_array
from climateeval.diags._utils import union
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from climateeval import Variable

SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "ann": tuple(range(1, 13)),
    "djf": (12, 1, 2),
    "mam": (3, 4, 5),
    "jja": (6, 7, 8),
    "son": (9, 10, 11),
}
"""Months of the supported seasons."""

_MIN_DAY_COVERAGE = 0.9
"""Fraction of the days of a season that must be present for it to count."""
_DAYS_PER_MONTH = 30.4
_DAILY_SPACING_MAX = 1.5


class PrecipitationStatistics(SimpleDiagnostic):
    """Manage precipitation frequency and intensity statistics of daily rain."""

    _final_coordinates = (Coordinate("time"),)
    _output_metrics_table: ClassVar[bool] = False

    # -- preprocessing -------------------------------------------------------

    def _safely_preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Preprocess the cube into the (time, statistic) table cube."""
        return self._preprocess(cube, variable)

    def _preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Compute the statistics of every season and year."""
        options = variable.diagnostic_kwargs
        seasons = [
            str(s).lower() for s in options.get("seasons", ("ann", "djf", "jja"))
        ]
        unknown = set(seasons) - set(SEASON_MONTHS)
        if unknown:
            msg = (
                f"unknown season(s) {sorted(unknown)}; choose from "
                f"{list(SEASON_MONTHS)}"
            )
            raise ValueError(msg)
        aggregation = str(options.get("aggregation", "year"))
        if aggregation not in ("year", "period"):
            msg = f"'aggregation' must be 'year' or 'period', got '{aggregation}'"
            raise ValueError(msg)
        percentiles = tuple(float(q) for q in options.get("percentiles", (10, 90)))
        edges = _physics.log_rain_bins(
            float(options.get("r_min", 0.03)),
            float(options.get("ratio", 1.07)),
            int(options.get("n_bins", 150)),
        )
        threshold = float(options.get("threshold", _physics.WET_DAY_THRESHOLD))

        cube = self._to_mm_per_day(cube)
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                options.get("grid", "1x1"),
                options.get("regrid_scheme", "area_weighted"),
                cache_weights=True,
            )

        results = self._statistics_by_group(
            cube,
            seasons,
            aggregation,
            lambda values, weights: _physics.precipitation_statistics(
                values,
                edges,
                threshold,
                percentiles,
                weights=weights,
            ),
        )
        wanted = options.get("statistics")
        return self._table_cube(cube, results, seasons, aggregation, wanted)

    @staticmethod
    def _to_mm_per_day(cube: Cube) -> Cube:
        """Convert a precipitation flux (kg m-2 s-1) to mm day-1 if necessary."""
        unit = str(cube.units)
        if unit in ("kg m-2 s-1", "kg m-2 s-1 "):
            cube = cube * 86400.0
            cube.units = "mm day-1"
        elif unit != "mm day-1":
            cube = cube.copy()
            cube.convert_units("mm day-1")
        return cube

    def _statistics_by_group(
        self,
        cube: Cube,
        seasons: list[str],
        aggregation: str,
        compute: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    ) -> dict[tuple[str, int], dict[str, float]]:
        """Compute the statistics of every (season, year), or of the period."""
        time = cube.coord("time")
        dates = time.units.num2date(time.points)
        years = np.array([d.year for d in dates])
        months = np.array([d.month for d in dates])
        if (
            time.points.size > 1
            and np.median(np.diff(time.points)) > _DAILY_SPACING_MAX
        ):
            msg = "PrecipitationStatistics needs daily data"
            raise ValueError(msg)
        lat = cube.coord("latitude").points.astype(float)
        weights_2d = np.cos(np.deg2rad(lat))[:, None]

        pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            s: [] for s in seasons
        }
        results: dict[tuple[str, int], dict[str, float]] = {}
        december: tuple[int, np.ndarray] | None = None  # (year, December block)
        for year in np.unique(years):
            index = np.flatnonzero(years == year)
            block = cube_to_array(
                cube[int(index[0]) : int(index[-1]) + 1],
                "time",
                "latitude",
                "longitude",
            )
            block_months = months[index]
            for season, sample in self._complete_samples(
                seasons,
                int(year),
                block,
                block_months,
                december,
            ):
                values, weights = self._flatten(sample, weights_2d)
                if aggregation == "year":
                    results[(season, int(year))] = compute(values, weights)
                else:
                    pooled[season].append((values, weights))
            december = (int(year), block[block_months == 12])  # noqa: PLR2004
        if aggregation == "period":
            for season, parts in pooled.items():
                if not parts:
                    continue
                values = np.concatenate([p[0] for p in parts])
                weights = np.concatenate([p[1] for p in parts])
                results[(season, int(years[0]))] = compute(values, weights)
        return results

    @staticmethod
    def _flatten(
        sample: np.ndarray,
        weights_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pool a (time, lat, lon) sample into valid values and their weights."""
        weights = np.broadcast_to(weights_2d, sample.shape)
        valid = np.isfinite(sample)
        return sample[valid].astype(np.float64), weights[valid].astype(np.float64)

    @staticmethod
    def _complete_samples(
        seasons: list[str],
        year: int,
        block: np.ndarray,
        block_months: np.ndarray,
        december: tuple[int, np.ndarray] | None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Yield the (season, days) of a year that are complete."""
        for season in seasons:
            months = SEASON_MONTHS[season]
            if season == "djf":
                if december is None or december[0] != year - 1:
                    continue
                in_year = np.isin(block_months, (1, 2))
                sample = np.concatenate([december[1], block[in_year]])
                present = set(np.unique(block_months[in_year]).tolist())
                if december[1].shape[0] > 0:
                    present.add(12)
            else:
                in_year = np.isin(block_months, months)
                sample = block[in_year]
                present = set(np.unique(block_months[in_year]).tolist())
            if present != set(months):
                continue
            if sample.shape[0] < _MIN_DAY_COVERAGE * _DAYS_PER_MONTH * len(months):
                continue
            yield season, sample

    # -- output ---------------------------------------------------------------

    def _table_cube(
        self,
        cube: Cube,
        results: dict[tuple[str, int], dict[str, float]],
        seasons: list[str],
        aggregation: str,
        wanted: list[str] | None,
    ) -> Cube:
        """Wrap the results in a (time, statistic) cube with named columns."""
        if not results:
            msg = "no complete season of daily data"
            raise ValueError(msg)
        keys = sorted({k for r in results.values() for k in r})
        if wanted is not None:
            keys = [k for k in keys if k in set(wanted)]
        columns = [(season, key) for season in seasons for key in keys]
        times = sorted({year for (_season, year) in results})
        data = np.full((len(times), len(columns)), np.nan)
        for (season, year), values in results.items():
            for j, (col_season, key) in enumerate(columns):
                if col_season == season and key in values:
                    data[times.index(year), j] = values[key]

        source = cube.coord("time")
        sample = source.units.num2date(source.points[:1])[0]
        points = [source.units.date2num(type(sample)(year, 7, 1, 12)) for year in times]
        time_coord = DimCoord(
            np.array(points, dtype=float),
            standard_name="time",
            long_name="time",
            var_name="time",
            units=source.units,
        )
        index_coord = DimCoord(
            np.arange(len(columns), dtype=float),
            long_name="statistic_index",
            var_name="statistic_index",
            units="1",
        )
        names = AuxCoord(
            np.array([f"{key}_{season}" for season, key in columns]),
            long_name="statistic",
            var_name="statistic",
        )
        table = Cube(
            data,
            var_name="precipitation_statistics",
            units="1",
            dim_coords_and_dims=[(time_coord, 0), (index_coord, 1)],
            aux_coords_and_dims=[(names, 1)],
        )
        with setup_esmvaltool_config_and_logging():
            return regrid_time(
                table,
                frequency="yr" if aggregation == "year" else "mon",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )

    def _get_raw_output_table(
        self,
        cube: Cube,
        *,
        variable: Variable,
        data_id: str,
        data_type: str,
    ) -> ibis.Table:
        """Get the table of a (time, statistic) cube: one column per statistic."""
        time = cube.coord("time")
        dates = time.units.num2date(time.points)
        frame = pd.DataFrame({"time": [str(d) for d in dates]})
        names = cube.coord("statistic").points
        for j, name in enumerate(names):
            frame[f"{variable.id}_{name}"] = np.asarray(cube.data[:, j], dtype=float)
        table = ibis.memtable(frame)
        casts = {
            c: ("timestamp(6)" if c == "time" else self._dtype) for c in table.columns
        }
        table = table.cast(**casts)
        table = table.mutate(**{self._data_id_column: ibis.literal(data_id)})
        return table.mutate(**{self._data_type_column: ibis.literal(data_type)})

    def _combine_raw_output_tables(
        self,
        *raw_output_tables: ibis.Table,
    ) -> ibis.Table | None:
        """Combine the tables of all variables and data sources."""
        if not raw_output_tables:
            return None
        table = union(*raw_output_tables)
        unique = (self._data_id_column, self._data_type_column, "time")
        value_columns = sorted(c for c in table.columns if c not in unique)
        table = table.group_by(unique).aggregate(
            **{c: table[c].first() for c in value_columns},
        )
        return table.select(*unique, *value_columns)
