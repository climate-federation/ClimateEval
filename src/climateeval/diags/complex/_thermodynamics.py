"""Provide MSU-equivalent temperature and inversion-strength diagnostics.

Both diagnostics combine several variables of one data set (a temperature
profile with surface fields), which the per-variable simple-diagnostic
machinery cannot do, so they are complex diagnostics of a single data key (by
default ``"data"``; set ``data_key`` to the experiment name used by the caller).
The same physics (:mod:`climateeval.diags._physics`) runs on a model and on a
reanalysis if the reanalysis cubes are passed to ``get_output`` in the same
way.

:class:`MSUTemperature`
    Monthly (or annual) MSU/AMSU-equivalent brightness temperatures of the
    lower (TLT) and mid (TMT) troposphere from ``ta`` on pressure levels,
    ``ts`` and ``ps``, as a tropical mean or on the grid.
:class:`InversionStrength`
    Monthly (or annual) lower-tropospheric stability and estimated inversion
    strength from ``ta`` at 700 hPa, a surface temperature, ``ps`` and,
    optionally, ``hurs``.

The output variables (``tlt``, ``tmt``, ``eis``, ``lts``) have to be defined in
``variables.yml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import annual_statistics, regrid
from iris.cube import Cube
from loguru import logger

from climateeval import Variable, VerticalLevel
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import get_prepared_cube
from climateeval.diags import _physics
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags._cube_arrays import cube_to_array
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._base import ComplexDiagnostic
from climateeval.diags.complex._scalar import calendar_years_and_months

if TYPE_CHECKING:
    from climateeval.data import ComplexDataSource

logger = logger.opt(colors=True)

_TIME_CHUNK = 60
"""Number of months processed at once (bounds the memory of the profiles)."""

_AGGREGATIONS = ("monthly", "annual")
_FULL_LONGITUDE = (0.0, 360.0)


class _GriddedThermodynamicDiagnostic(ComplexDiagnostic):
    """Manage diagnostics deriving gridded monthly fields from several variables."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        *,
        data_key: str = "data",
        spatial: str = "gridded",
        lat_range: tuple[float, float] = (-20.0, 20.0),
        aggregation: str = "monthly",
        grid: str = DEFAULT_GRID,
        landsea_mask: str = "*",
        **kwargs: Any,
    ) -> None:
        """Initialize class instance.

        Parameters
        ----------
        name:
            Diagnostic name.
        data_key:
            Key of the data dictionary holding the cubes.
        spatial:
            ``"gridded"`` for fields on the grid or ``"mean"`` for the
            area mean over ``lat_range`` (all longitudes).
        lat_range:
            South and north limit (degrees) of the area mean.
        aggregation:
            ``"monthly"`` or ``"annual"`` (annual means of the monthly values).
        grid:
            Target grid of the regridding.
        landsea_mask:
            ``"*"`` (default), ``"land_only"`` or ``"sea_only"``, applied to
            all input variables.
        **kwargs:
            Passed to :class:`ComplexDiagnostic` (``data_root_dir`` etc.).

        """
        if spatial not in ("gridded", "mean"):
            msg = f"'spatial' must be 'gridded' or 'mean', got '{spatial}'"
            raise ValueError(msg)
        if aggregation not in _AGGREGATIONS:
            msg = f"'aggregation' must be one of {_AGGREGATIONS}, got '{aggregation}'"
            raise ValueError(msg)
        super().__init__(name, **kwargs)
        self._required_data_keys = (data_key,)
        self._data_key = data_key
        self._spatial = spatial
        self._lat_range = (float(lat_range[0]), float(lat_range[1]))
        self._aggregation = aggregation
        self._grid = grid
        self._landsea_mask = landsea_mask

    @property
    def _output_names(self) -> tuple[str, ...]:
        """Get the names (and column names) of the output variables."""
        raise NotImplementedError

    def _postprocess_output(self, output: DiagnosticOutput) -> DiagnosticOutput:
        """Merge the per-variable tables into one table and order the columns.

        One row per ``(data_id, data_type, time[, latitude, longitude])`` with
        one column per output variable, as in the other time-series outputs.
        """
        if output.raw_output is None:
            return output
        coordinates = ["time"]
        if self._spatial == "gridded":
            coordinates += ["latitude", "longitude"]
        unique_columns = (self._data_id_column, self._data_type_column, *coordinates)
        raw_output = self._remove_superfluous_rows(
            output.raw_output,
            unique_columns=unique_columns,
            variable_columns=self._output_names,
        )
        raw_output = raw_output.select(*unique_columns, *self._output_names)
        return DiagnosticOutput(
            raw_output=raw_output,
            metrics=output.metrics,
            variables=output.variables,
            data_sources=output.data_sources,
        )

    # -- cube handling -----------------------------------------------------

    def _variable(self, var_name: str, **kwargs: Any) -> Variable:
        """Get an input variable with this diagnostic's masking."""
        return Variable(
            var_name,
            var_name,
            "mon",
            landsea_mask=self._landsea_mask,
            **kwargs,
        )

    def _regridded_cube(self, data: Any, variable: Variable) -> Cube:
        """Get one input variable on the target grid."""
        cube = get_prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            return regrid(cube, self._grid, "linear", cache_weights=True)

    @staticmethod
    def _align_time(cubes: dict[str, Cube]) -> tuple[dict[str, Cube], Cube]:
        """Restrict cubes to the months they have in common.

        Returns the aligned cubes and the first of them (for its coordinates).
        """

        def keys(cube: Cube) -> np.ndarray:
            years, months = calendar_years_and_months(cube)
            return years * 12 + months

        common: np.ndarray | None = None
        for cube in cubes.values():
            common = (
                keys(cube) if common is None else np.intersect1d(common, keys(cube))
            )
        if common is None or common.size == 0:
            msg = "the input variables share no month"
            raise ValueError(msg)
        aligned = {
            name: cube[np.flatnonzero(np.isin(keys(cube), common))]
            for name, cube in cubes.items()
        }
        return aligned, next(iter(aligned.values()))

    def _output_cube(
        self,
        values: np.ndarray,
        template: Cube,
        *,
        var_name: str,
        units: str,
    ) -> Cube:
        """Wrap ``(time, lat, lon)`` values (or ``(time,)`` for a mean) in a cube."""
        time_coord = template.coord("time").copy()
        if self._spatial == "mean":
            cube = Cube(
                values,
                var_name=var_name,
                units=units,
                dim_coords_and_dims=[(time_coord, 0)],
            )
        else:
            cube = Cube(
                values,
                var_name=var_name,
                units=units,
                dim_coords_and_dims=[
                    (time_coord, 0),
                    (template.coord("latitude").copy(), 1),
                    (template.coord("longitude").copy(), 2),
                ],
            )
        if self._aggregation == "annual":
            with setup_esmvaltool_config_and_logging():
                cube = annual_statistics(cube, "mean")
        return cube

    def _reduce(self, field: np.ndarray, template: Cube) -> np.ndarray:
        """Area-mean a ``(time, lat, lon)`` field if the output is a mean."""
        if self._spatial != "mean":
            return field
        return _physics.box_mean(
            field,
            template.coord("latitude").points.astype(float),
            template.coord("longitude").points.astype(float),
            self._lat_range,
            _FULL_LONGITUDE,
        )


class MSUTemperature(_GriddedThermodynamicDiagnostic):
    """Manage MSU/AMSU-equivalent temperature diagnostics (TLT, TMT).

    Computes, for every month, the brightness temperature that the MSU/AMSU
    channels would measure from the model's temperature profile with the RSS
    static weighting functions
    (:func:`~climateeval.diags._physics.msu_weighted_temperature`), truncated
    at the surface pressure and renormalised. Land and ocean weights are
    blended with the land fraction ``sftlf`` if the data contain it, ocean
    weights are used otherwise.

    Required data (key ``data_key``, monthly): ``ta`` on pressure levels
    (masked below the surface is fine), ``ts`` and ``ps``; ``sftlf`` is
    optional.

    Keyword arguments (besides those of :class:`ComplexDiagnostic`):
    ``channels`` (default ``("tlt", "tmt")``; ``"tts"`` and ``"tls"`` also
    work), ``data_key`` (``"data"``), ``spatial`` (``"mean"`` default here, or
    ``"gridded"``), ``lat_range`` (``(-20, 20)``), ``aggregation``
    (``"monthly"`` or ``"annual"``), ``grid`` and ``landsea_mask``.

    Output (raw output table: ``data_id``, ``data_type``, ``time`` and, for
    ``spatial="gridded"``, ``lat`` and ``lon``): one column per channel,
    named after the channel (``tlt``, ``tmt``, ...), in K.
    """

    def __init__(
        self,
        name: str,
        *,
        channels: tuple[str, ...] = ("tlt", "tmt"),
        spatial: str = "mean",
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        unknown = set(channels) - set(_physics.MSU_CHANNELS)
        if unknown:
            msg = (
                f"unknown channel(s) {sorted(unknown)}; choose from "
                f"{_physics.MSU_CHANNELS}"
            )
            raise ValueError(msg)
        super().__init__(name, spatial=spatial, **kwargs)
        self._channels = tuple(channels)

    @property
    def _output_names(self) -> tuple[str, ...]:
        """Get the channels."""
        return self._channels

    def _land_fraction(self, data: Any) -> np.ndarray | None:
        """Get the land fraction (0-1) on the grid, None if not available."""
        try:
            cube = self._regridded_cube(
                data,
                Variable("sftlf", "sftlf", "fx", landsea_mask=self._landsea_mask),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"No land fraction for MSU weighting, using ocean ({exc})")
            return None
        return np.clip(cube_to_array(cube, "latitude", "longitude") / 100.0, 0.0, 1.0)

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate MSU-equivalent temperatures."""
        data = complex_data_source.data[self._data_key]
        ta_cube = self._regridded_cube(data, self._variable("ta"))
        cubes, template = self._align_time(
            {
                "ta": ta_cube,
                "ts": self._regridded_cube(data, self._variable("ts")),
                "ps": self._regridded_cube(data, self._variable("ps")),
            },
        )
        plev_coord = cubes["ta"].coord("air_pressure")
        plev = plev_coord.units.convert(plev_coord.points.astype(float), "Pa")
        ta = cube_to_array(cubes["ta"], "air_pressure", "time", "latitude", "longitude")
        ts = cube_to_array(cubes["ts"], "time", "latitude", "longitude")
        ps = cube_to_array(cubes["ps"], "time", "latitude", "longitude")
        land = self._land_fraction(data)
        lat = template.coord("latitude").points.astype(float)
        rows = np.ones(lat.size, dtype=bool)
        if self._spatial == "mean":
            rows = (lat >= self._lat_range[0]) & (lat <= self._lat_range[1])

        output: dict[Variable, Cube] = {}
        for channel in self._channels:
            field = np.full(ts.shape, np.nan)
            for start in range(0, ts.shape[0], _TIME_CHUNK):
                chunk = slice(start, start + _TIME_CHUNK)
                land_chunk = (
                    None
                    if land is None
                    else np.broadcast_to(land[rows], ts[chunk][:, rows].shape)
                )
                field[chunk][:, rows] = _physics.msu_weighted_temperature(
                    ta[:, chunk][:, :, rows],
                    plev,
                    ts[chunk][:, rows],
                    ps[chunk][:, rows],
                    channel,
                    land_fraction=land_chunk,
                )
            variable = Variable(channel, channel, "mon")
            output[variable] = self._output_cube(
                self._reduce(field, template),
                template,
                var_name=channel,
                units="K",
            )
        return output


class InversionStrength(_GriddedThermodynamicDiagnostic):
    """Manage lower-tropospheric stability and estimated inversion strength.

    ``LTS = theta_700 - theta_sfc`` and ``EIS = LTS - Gamma_m^850 (z_700 -
    z_LCL)`` (Wood and Bretherton, 2006) per month
    (:func:`~climateeval.diags._physics.lower_tropospheric_stability`,
    :func:`~climateeval.diags._physics.estimated_inversion_strength`).

    Required data (key ``data_key``, monthly): ``ta`` on pressure levels
    (interpolated to 700 hPa), the surface temperature variable
    ``surface_temperature`` (``"ts"`` by default; ``"tas"`` is the usual
    choice for observational estimates) and ``ps``. ``hurs`` is used for the
    lifting condensation level if the data contain it, otherwise a relative
    humidity of 80 % is assumed.

    Keyword arguments (besides those of :class:`ComplexDiagnostic`):
    ``surface_temperature``, ``outputs`` (default ``("eis", "lts")``),
    ``data_key`` (``"data"``), ``spatial`` (``"gridded"`` default, or
    ``"mean"``), ``lat_range``, ``aggregation``, ``grid`` and ``landsea_mask``
    (``"sea_only"`` is the usual choice).

    Output (raw output table: ``data_id``, ``data_type``, ``time`` and, for
    ``spatial="gridded"``, ``lat`` and ``lon``): columns ``eis`` and ``lts``
    (as requested), in K.
    """

    _OUTPUTS: ClassVar[tuple[str, ...]] = ("eis", "lts")

    def __init__(
        self,
        name: str,
        *,
        surface_temperature: str = "ts",
        outputs: tuple[str, ...] = ("eis", "lts"),
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        unknown = set(outputs) - set(self._OUTPUTS)
        if unknown:
            msg = f"unknown output(s) {sorted(unknown)}; choose from {self._OUTPUTS}"
            raise ValueError(msg)
        super().__init__(name, **kwargs)
        self._surface_temperature = surface_temperature
        self._outputs = tuple(outputs)

    @property
    def _output_names(self) -> tuple[str, ...]:
        """Get the requested outputs."""
        return self._outputs

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate LTS and EIS."""
        data = complex_data_source.data[self._data_key]
        level = VerticalLevel("air_pressure", "Pa", 70000.0)
        inputs = {
            "t700": self._regridded_cube(
                data, self._variable("ta", vertical_level=level),
            ),
            "tsfc": self._regridded_cube(
                data, self._variable(self._surface_temperature),
            ),
            "ps": self._regridded_cube(data, self._variable("ps")),
        }
        try:
            inputs["rh"] = self._regridded_cube(data, self._variable("hurs"))
        except Exception as exc:  # noqa: BLE001
            logger.info(f"No hurs for the EIS lifting condensation level ({exc})")
        cubes, template = self._align_time(inputs)
        arrays = {
            name: cube_to_array(cube, "time", "latitude", "longitude")
            for name, cube in cubes.items()
        }
        fields = {
            "lts": _physics.lower_tropospheric_stability(
                arrays["t700"],
                arrays["tsfc"],
                arrays["ps"],
            ),
            "eis": _physics.estimated_inversion_strength(
                arrays["t700"],
                arrays["tsfc"],
                arrays["ps"],
                rh_sfc=arrays.get("rh"),
            ),
        }
        return {
            Variable(name, name, "mon"): self._output_cube(
                self._reduce(fields[name], template),
                template,
                var_name=name,
                units="K",
            )
            for name in self._outputs
        }
