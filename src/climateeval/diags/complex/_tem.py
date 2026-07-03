"""Provide Transformed Eulerian Mean (TEM) and diabatic velocity diagnostics.

Three quantities are computed from zonal-mean, climatological 3-D fields:

psi_star : TEM residual mass streamfunction [kg s⁻¹]
    Ψ*(φ, p) = (2π a cosφ / g) ∫₀ᵖ v*(p′) dp′
    where the integral runs from the top of the atmosphere downward.
    When v'T' (vt) is supplied the full TEM correction is applied; otherwise
    the Eulerian-mean streamfunction (v* = v̄) is returned.

v_star / w_star : TEM residual velocities [m s⁻¹ / Pa s⁻¹]
    v*(φ, p) = v̄ + ∂/∂p [v′θ′ / (∂θ̄/∂p)]
    ω*(φ, p) = ω̄ − (1/(a cosφ)) ∂/∂φ [cosφ v′θ′ / (∂θ̄/∂p)]
    Without vt the eddy term vanishes and the Eulerian means are returned.

w_dia : Diabatic vertical velocity inferred from thermodynamic residual [Pa s⁻¹]
    In the annual-mean steady state the thermodynamic equation gives the
    diabatic potential-temperature tendency:
        Q_θ = v̄ (∂θ̄ / a ∂φ) + ω̄ (∂θ̄/∂p)
    The cross-isentropic (diabatic) omega is then:
        ω_dia = Q_θ / (∂θ̄/∂p)
    In the stratosphere, where latent/sensible heating is negligible, this
    approximates the radiatively-inferred diabatic velocity.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    climate_statistics,
    extract_levels,
    regrid,
    zonal_statistics,
)
from iris import NameConstraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from loguru import logger
from scipy.integrate import cumulative_trapezoid

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import get_prepared_cube
from climateeval.data import ComplexDataSource
from climateeval.diags._utils import DEFAULT_GRID, PLEV19_LEVELS
from climateeval.diags.complex._base import ComplexDiagnostic

logger = logger.opt(colors=True)

# Physical constants
_R_DRY = 287.058  # J kg⁻¹ K⁻¹
_CP = 1004.709  # J kg⁻¹ K⁻¹
_KAPPA = _R_DRY / _CP
_P0 = 100000.0  # Pa
_EARTH_RADIUS = 6.371e6  # m
_G = 9.80665  # m s⁻²
_TWO_PI = 2.0 * np.pi


def _preprocess_to_zonal_clim(cube: Cube) -> Cube:
    """Regrid → extract PLEV19 levels → zonal mean → climatology."""
    with setup_esmvaltool_config_and_logging():
        cube = extract_levels(cube, PLEV19_LEVELS, "linear", coordinate="air_pressure")
        cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
        cube = zonal_statistics(cube, "mean")
        cube = climate_statistics(cube, "mean", "full")
    return cube


def _make_zonal_clim_cube(
    data: np.ndarray,
    pressure: np.ndarray,
    latitude: np.ndarray,
    var_name: str,
    units: str,
) -> Cube:
    """Create a (plev, lat) iris Cube with air_pressure and latitude coordinates."""
    plev_coord = DimCoord(
        pressure.astype(np.float32),
        standard_name="air_pressure",
        units="Pa",
    )
    lat_coord = DimCoord(
        latitude.astype(np.float32),
        standard_name="latitude",
        units="degrees_north",
    )
    cube = Cube(
        data.astype(np.float32),
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(plev_coord, 0), (lat_coord, 1)],
    )
    return cube


class TEMDiagnostic(ComplexDiagnostic):
    """Compute TEM streamfunction, residual velocities, and diabatic velocity.

    Required data key
    -----------------
    ``"data"`` : CubeList or xarray Dataset
        Must contain ``va`` (meridional wind) and ``ta`` (temperature) for all
        outputs, plus ``wap`` (omega) for the residual velocities and diabatic
        velocity.  ``vt`` (northward heat flux, v′T′) is optional; when absent
        the eddy correction term is set to zero.

    Outputs (all on PLEV19 pressure × 2° latitude grid)
    ----------------------------------------------------
    psi_star : TEM residual mass streamfunction [kg s⁻¹]
    v_star   : TEM residual meridional velocity  [m s⁻¹]
    w_star   : TEM residual vertical velocity    [Pa s⁻¹]
    w_dia    : Diabatic omega (thermodynamic residual) [Pa s⁻¹]
    """

    _required_data_keys: ClassVar[tuple[str, ...]] = ("data",)

    # Input variables
    _va: ClassVar[Variable] = Variable("va", "va", "mon")
    _ta: ClassVar[Variable] = Variable("ta", "ta", "mon")
    _wap: ClassVar[Variable] = Variable("wap", "wap", "mon")
    _vt: ClassVar[Variable] = Variable("vt", "vt", "mon")

    # Output variables
    _psi_star: ClassVar[Variable] = Variable("psi_star", "psi_star", "mon")
    _v_star: ClassVar[Variable] = Variable("v_star", "v_star", "mon")
    _w_star: ClassVar[Variable] = Variable("w_star", "w_star", "mon")
    _w_dia: ClassVar[Variable] = Variable("w_dia", "w_dia", "mon")

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Compute TEM quantities from zonal-mean climatological fields."""
        cubelist = complex_data_source.data["data"]

        # --- Preprocess inputs to (plev, lat) climatological zonal means -----
        va_cube = _preprocess_to_zonal_clim(get_prepared_cube(cubelist, self._va))
        ta_cube = _preprocess_to_zonal_clim(get_prepared_cube(cubelist, self._ta))

        has_wap = self._has_variable(cubelist, self._wap)
        wap_cube = (
            _preprocess_to_zonal_clim(get_prepared_cube(cubelist, self._wap))
            if has_wap
            else None
        )

        has_vt = self._has_variable(cubelist, self._vt)
        vt_cube = (
            _preprocess_to_zonal_clim(get_prepared_cube(cubelist, self._vt))
            if has_vt
            else None
        )

        if not has_wap:
            logger.warning("wap not found: w_star and w_dia will be set to NaN")

        # --- Extract numpy arrays ----------------------------------------
        p = va_cube.coord("air_pressure").points.astype(float)  # (nlev,)
        lat_deg = va_cube.coord("latitude").points.astype(float)  # (nlat,)
        phi = np.deg2rad(lat_deg)  # (nlat,)
        cos_phi = np.cos(phi)  # (nlat,)

        va_bar = va_cube.data.astype(float)  # (nlev, nlat)
        ta_bar = ta_cube.data.astype(float)  # (nlev, nlat)
        wap_bar = (
            wap_cube.data.astype(float) if has_wap else np.full_like(va_bar, np.nan)
        )
        vt_bar = vt_cube.data.astype(float) if has_vt else None

        # --- Potential temperature θ̄ = T̄ (p₀/p)^κ -------------------------
        kappa_factor = (
            _P0 / p[:, np.newaxis]
        ) ** _KAPPA  # (nlev, 1) broadcast → (nlev, nlat)
        theta_bar = ta_bar * kappa_factor

        # --- ∂θ̄/∂p using finite differences --------------------------------
        # p[0] = 100000 Pa (surface), p[-1] = 100 Pa (top) → p is decreasing
        dtheta_dp = np.gradient(theta_bar, p, axis=0)

        # Guard against near-zero stability (avoid division blow-up)
        stability_min = np.percentile(np.abs(dtheta_dp[np.isfinite(dtheta_dp)]), 1)
        safe_dtheta_dp = np.where(
            np.abs(dtheta_dp) < stability_min,
            np.sign(dtheta_dp + 1e-30) * stability_min,
            dtheta_dp,
        )

        # --- Eddy flux term: v′θ′ / (∂θ̄/∂p) --------------------------------
        if vt_bar is not None:
            # Convert v′T′ → v′θ′ via the Exner factor
            vprime_thetaprime = vt_bar * kappa_factor  # K m s⁻¹ → (θ-scaled)
            eddy_ratio = vprime_thetaprime / safe_dtheta_dp  # m s⁻¹
            logger.debug("TEMDiagnostic: using v′T′ for full TEM correction")
        else:
            eddy_ratio = np.zeros_like(va_bar)
            logger.debug(
                "TEMDiagnostic: vt absent, returning Eulerian-mean circulation",
            )

        # --- Residual velocities -------------------------------------------
        # v* = v̄ + ∂(eddy_ratio)/∂p
        v_star = va_bar + np.gradient(eddy_ratio, p, axis=0)

        # ω* = ω̄ − (1/(a cosφ)) ∂(cosφ eddy_ratio)/∂φ
        cos_eddy = cos_phi[np.newaxis, :] * eddy_ratio  # (nlev, nlat)
        d_cos_eddy_dphi = np.gradient(cos_eddy, phi, axis=1)  # ∂/∂φ
        with np.errstate(invalid="ignore", divide="ignore"):
            w_star = wap_bar - d_cos_eddy_dphi / (
                _EARTH_RADIUS * cos_phi[np.newaxis, :]
            )
        # Mask poles where cosφ → 0
        w_star[:, cos_phi < 1e-3] = np.nan

        # --- TEM streamfunction Ψ* ----------------------------------------
        # Ψ*(φ, p) = (2π a cosφ / g) ∫₀ᵖ v* dp′
        # Integrate from small p (top) to each p level (downward).
        # PLEV19 levels run from p[0]=100000 (bottom) to p[-1]=100 (top),
        # so we reverse to ascending-p order, integrate, then reverse back.
        idx_asc = np.argsort(p)  # 100 Pa first (top)
        p_asc = p[idx_asc]
        v_asc = v_star[idx_asc, :]
        # cumulative_trapezoid(initial=0) → shape (nlev, nlat)
        psi_asc = cumulative_trapezoid(v_asc, p_asc, axis=0, initial=0)
        # Restore to original level order
        idx_restore = np.argsort(idx_asc)
        psi_star = psi_asc[idx_restore, :] * (
            _TWO_PI * _EARTH_RADIUS * cos_phi[np.newaxis, :] / _G
        )
        # Mask poles
        psi_star[:, cos_phi < 1e-3] = np.nan

        # --- Diabatic omega (thermodynamic residual) ----------------------
        # In annual-mean steady state: Q_θ = v̄ ∂θ̄/(a∂φ) + ω̄ ∂θ̄/∂p
        # Cross-isentropic velocity: ω_dia = Q_θ / (∂θ̄/∂p)
        dtheta_dphi = np.gradient(theta_bar, phi, axis=1)  # ∂θ̄/∂φ
        Q_theta = va_bar * dtheta_dphi / _EARTH_RADIUS + wap_bar * dtheta_dp
        with np.errstate(invalid="ignore", divide="ignore"):
            w_dia = Q_theta / safe_dtheta_dp
        # Mask where stability is too weak (unphysical)
        w_dia[np.abs(dtheta_dp) < stability_min] = np.nan

        # --- Build output iris Cubes ---------------------------------------
        psi_cube = _make_zonal_clim_cube(psi_star, p, lat_deg, "psi_star", "kg s-1")
        v_star_cube = _make_zonal_clim_cube(v_star, p, lat_deg, "v_star", "m s-1")
        w_star_cube = _make_zonal_clim_cube(w_star, p, lat_deg, "w_star", "Pa s-1")
        w_dia_cube = _make_zonal_clim_cube(w_dia, p, lat_deg, "w_dia", "Pa s-1")

        return {
            self._psi_star: psi_cube,
            self._v_star: v_star_cube,
            self._w_star: w_star_cube,
            self._w_dia: w_dia_cube,
        }

    def _postprocess_output(self, output):
        """Merge per-variable rows into a single (plev, lat) table."""
        from climateeval.diags._base import DiagnosticOutput

        if output.raw_output is None:
            return output

        unique_cols = (
            self._data_id_column,
            self._data_type_column,
            "air_pressure",
            "latitude",
        )
        var_cols = (
            self._psi_star.id,
            self._v_star.id,
            self._w_star.id,
            self._w_dia.id,
        )
        raw_output = self._remove_superfluous_rows(
            output.raw_output,
            unique_columns=unique_cols,
            variable_columns=var_cols,
        )
        raw_output = raw_output.select((*unique_cols, *var_cols))
        return DiagnosticOutput(
            raw_output=raw_output,
            metrics=output.metrics,
            variables=output.variables,
            data_sources=output.data_sources,
        )

    @staticmethod
    def _has_variable(cubelist: CubeList, variable: Variable) -> bool:
        """Return True if var_name is present in cubelist."""
        try:
            cubelist.extract_cube(NameConstraint(var_name=variable.var_name))
            return True
        except Exception:  # noqa: BLE001
            return False
