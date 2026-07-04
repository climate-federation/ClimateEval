"""Static (matplotlib) figure builders for publication-ready PNG/PDF export.

Mirrors the Plotly ``_FIGURE_BUILDERS`` registry in ``_figures.py`` with a
matplotlib backend styled after the ESMValTool ``monitor/multi_datasets.py``
diagnostic as configured by the ICONEval ``recipe_templates`` (seaborn "ticks"
theme, 7×5 inch figures, subtle gray gridlines, legend centred below the axes,
black reference lines with per-reference linestyles).

Builders share the Plotly builders' signature —
``build(data_df, variables_df, metrics_df=None) -> dict[var_id, Figure]`` —
so ``generate_static_plots()`` can reuse the same database reading and
diagnostic-type detection as the interactive report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure

from matplotlib.ticker import MaxNLocator

from climateeval.report._common import (
    _combine_pivots,
    _data_range,
    _get_ref_datasets,
    _sort_datasets,
    _var_label,
)
from climateeval.report._db import detect_diag_type, get_var_cols, read_database
from climateeval.report._style import (
    _SEABORN_COLORS,
    _bias_colorscale_for,
    _colorscale_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import matplotlib.axes

# ---------------------------------------------------------------------------
# ESMValTool/ICONEval style constants (monitor/multi_datasets.py defaults +
# ICONEval recipe_templates plot kwargs)
# ---------------------------------------------------------------------------

_FIGSIZE = (7.0, 5.0)  # figure_kwargs.figsize in every ICONEval recipe
_SAVEFIG_KWARGS = {"bbox_inches": "tight", "orientation": "landscape"}
_GRIDLINE_KWARGS = {"color": "darkgray", "linewidth": 0.5, "alpha": 0.3}
_LEGEND_KWARGS = {"loc": "upper center", "borderaxespad": 0.0}

# Line styling per data_type, following the ICONEval timeseries/annual-cycle
# plot_kwargs: the benchmarked model drawn on top, observations/references in
# black (linestyle distinguishes multiple references), auxiliary datasets
# (e.g. a CMIP6 ensemble) as a thin light-gray background bundle.
_BENCHMARK_KWARGS = {"linewidth": 1.25, "zorder": 2.5}
_REFERENCE_KWARGS = {"color": "black", "linewidth": 1.0, "zorder": 2.4}
_OTHER_KWARGS = {"color": "lightgray", "linewidth": 0.75, "zorder": 1.0}
_REFERENCE_LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]

_SEABORN_STYLE = "ticks"  # multi_datasets.py seaborn_settings default


def _new_figure() -> tuple[Figure, matplotlib.axes.Axes]:
    """Return an ICONEval-styled (figure, axes) pair (no pyplot state)."""
    with sns.axes_style(_SEABORN_STYLE):
        fig = Figure(figsize=_FIGSIZE)
        axes = fig.add_subplot(111)
    return fig, axes


def _line_kwargs(dtype: str, color_idx: int, ref_idx: int) -> dict:
    """Return matplotlib plot kwargs for a dataset line."""
    if dtype == "reference":
        return {
            **_REFERENCE_KWARGS,
            "linestyle": _REFERENCE_LINESTYLES[ref_idx % len(_REFERENCE_LINESTYLES)],
        }
    if dtype == "to_benchmark":
        return {
            **_BENCHMARK_KWARGS,
            "color": _SEABORN_COLORS[color_idx % len(_SEABORN_COLORS)],
        }
    return dict(_OTHER_KWARGS)


def _finish_line_axes(
    axes: matplotlib.axes.Axes,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "",
) -> None:
    """Apply the shared ICONEval line-plot decoration (grid, labels, legend)."""
    axes.grid(visible=True, **_GRIDLINE_KWARGS)
    axes.set_axisbelow(True)
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    if xlabel:
        axes.set_xlabel(xlabel)
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(
            handles,
            labels,
            **_LEGEND_KWARGS,
            bbox_to_anchor=(0.5, -0.12),
            ncols=min(len(handles), 3),
        )


def _plot_reasonable_range(
    axes: matplotlib.axes.Axes,
    variables_df: pd.DataFrame,
    var_id: str,
) -> None:
    """Draw dashed reasonable_min/reasonable_max guide lines, if defined."""
    row = variables_df[variables_df["id"] == var_id]
    if row.empty:
        return
    for col in ("reasonable_min", "reasonable_max"):
        if col not in row:
            continue
        val = row[col].array[0]
        if val is None or np.isnan(val):
            continue
        axes.axhline(y=val, color="black", linestyle="dashed", linewidth=0.75)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def build_static_timeseries_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Time series, one figure per variable, all datasets overlaid.

    Follows the ICONEval ``recipe_basics_timeseries.yml`` look: single axes,
    benchmarked model in colour on top, references in black, auxiliary
    datasets as a light-gray bundle, legend below the axes.
    """
    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets = _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        fig, axes = _new_figure()
        _plot_reasonable_range(axes, variables_df, var_id)

        # Use markers when there are too few points for a visible line.
        marker = "o" if data_df["time"].nunique() < 3 else None

        color_idx = 0
        for dataset in all_datasets:
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            sub = grp[["time", var_id]].sort_values("time")
            if sub.empty:
                continue
            dtype = grp["data_type"].iloc[0]
            ref_idx = ref_datasets.index(dataset) if dataset in ref_datasets else 0
            kwargs = _line_kwargs(dtype, color_idx, ref_idx)
            axes.plot(
                sub["time"],
                sub[var_id],
                label=None if dtype == "other" else str(dataset),
                marker=marker,
                **kwargs,
            )
            if dtype == "to_benchmark":
                color_idx += 1

        ylabel = f"{var_id} [{units}]" if units else var_id
        _finish_line_axes(axes, title=long_name, ylabel=ylabel)
        figures[var_id] = fig

    return figures


# ---------------------------------------------------------------------------
# Map builder (cartopy)
# ---------------------------------------------------------------------------

# ESMValTool monitor "map" defaults: Robinson projection with
# central_longitude 10, horizontal colorbars, ICONEval fontsize 8 and
# 12 discrete contourf levels; stats text at x=0.0 (mean) / x=0.92 (metrics).
_MAP_FONTSIZE = 8
_MAP_STATS_FONTSIZE = 6
_MAP_LEVELS = 12
_MAP_CBAR_ORIENTATION = "horizontal"
_MAP_CBAR_ASPECT = 30.0


def _mpl_cmap(name: str) -> str:
    """Translate a Plotly colorscale name to a matplotlib colormap name."""
    import matplotlib as mpl

    if name in mpl.colormaps:
        return name
    return name.lower()


def _map_pivot(
    data_df: pd.DataFrame,
    dataset: str,
    var_id: str,
) -> pd.DataFrame | None:
    """Return a lat×lon pivot for one dataset (longitudes in [-180, 180])."""
    sub = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
    sv = sub[["latitude", "longitude", var_id]].copy()
    if sv.empty:
        return None
    sv["lon"] = sv["longitude"].apply(lambda x: x - 360 if x > 180 else x)
    p = (
        sv.groupby(["latitude", "lon"], as_index=False)[var_id]
        .mean()
        .pivot(index="latitude", columns="lon", values=var_id)
        .sort_index(axis=1)
    )
    return p if not p.empty else None


def _map_levels(vmin: float, vmax: float) -> np.ndarray:
    """Return ~_MAP_LEVELS nice contour boundaries covering [vmin, vmax]."""
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return MaxNLocator(nbins=_MAP_LEVELS).tick_values(vmin, vmax)


def _weighted_mean(pivot: pd.DataFrame) -> float:
    """Area (cos-latitude) weighted mean of a lat×lon pivot."""
    weights = np.cos(np.deg2rad(pivot.index.to_numpy(dtype=float)))
    vals = pivot.to_numpy(dtype=float)
    mask = np.isfinite(vals)
    w2d = np.broadcast_to(weights[:, None], vals.shape)
    wsum = (w2d * mask).sum()
    if wsum == 0:
        return float("nan")
    return float(np.nansum(vals * w2d * mask) / wsum)


def _weighted_rmse(bias: pd.DataFrame) -> float:
    """Area-weighted RMSE from a bias (model − reference) pivot."""
    sq = bias**2
    mean_sq = _weighted_mean(sq)
    return float(np.sqrt(mean_sq)) if np.isfinite(mean_sq) else float("nan")


def _weighted_r2(model: pd.DataFrame, ref: pd.DataFrame) -> float:
    """Area-weighted R² between two aligned lat×lon pivots.

    Same definition as multi_datasets.py (sklearn r2_score with area weights).
    """
    m, r = model.align(ref)
    vals_m = m.to_numpy(dtype=float).ravel()
    vals_r = r.to_numpy(dtype=float).ravel()
    lats = m.index.to_numpy(dtype=float)
    w = np.broadcast_to(
        np.cos(np.deg2rad(lats))[:, None],
        m.shape,
    ).ravel()
    mask = np.isfinite(vals_m) & np.isfinite(vals_r)
    if not mask.any():
        return float("nan")
    vals_m, vals_r, w = vals_m[mask], vals_r[mask], w[mask]
    ref_mean = np.average(vals_r, weights=w)
    ss_res = np.average((vals_r - vals_m) ** 2, weights=w)
    ss_tot = np.average((vals_r - ref_mean) ** 2, weights=w)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _fmt_stat(value: float, units: str) -> str:
    if not np.isfinite(value):
        return "—"
    formatted = f"{value:.2f}" if abs(value) >= 0.1 else f"{value:.2e}"  # noqa: PLR2004
    return f"{formatted} {units}".rstrip()


def _stats_text(
    axes: matplotlib.axes.Axes,
    x_pos: float,
    y_pos: float,
    text: str,
) -> None:
    """Small stats annotation inside an axes (multi_datasets.py style)."""
    axes.text(
        x_pos,
        y_pos,
        text,
        fontsize=_MAP_STATS_FONTSIZE,
        transform=axes.transAxes,
    )


def _annotate_bias_panel(  # noqa: PLR0913
    stats_text: Callable,
    axes: matplotlib.axes.Axes,
    *,
    bias: pd.DataFrame,
    model_pivot: pd.DataFrame,
    ref_pivot: pd.DataFrame,
    units: str,
    metrics: dict[str, float],
    x_avg: float = 0.0,
    x_bias: float = 0.92,
) -> None:
    """Add multi_datasets.py-style stats to a bias panel (mean, RMSE, R²).

    Pivots must have latitude as their index (transpose profile pivots).
    """
    stats_text(axes, x_avg, 0.95, _fmt_stat(_weighted_mean(bias), units))
    rmse = metrics.get("weighted_rmse", _weighted_rmse(bias))
    if np.isfinite(rmse):
        stats_text(axes, x_bias, 0.95, f"RMSE={_fmt_stat(rmse, units)}")
    r2 = _weighted_r2(model_pivot, ref_pivot)
    if np.isfinite(r2):
        stats_text(axes, x_bias, 0.85, f"R²={r2:.2f}")


def _map_metrics_for(
    metrics_df: pd.DataFrame | None,
    var_id: str,
    dataset: str,
) -> dict[str, float]:
    """Return {metric_name: value} for one (variable, dataset), if recorded."""
    if metrics_df is None or metrics_df.empty:
        return {}
    df = metrics_df
    if "var_id" in df.columns:
        df = df[df["var_id"] == var_id]
    if "data_id" in df.columns:
        df = df[df["data_id"] == dataset]
    if df.empty:
        return {}
    out = {}
    for col in ("weighted_rmse", "weighted_pearsonr"):
        if col in df.columns and pd.notna(df[col].iloc[0]):
            out[col] = float(df[col].iloc[0])
    return out


def build_static_map_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Global maps on a Robinson projection, ESMValTool monitor style.

    With a reference and exactly one benchmarked dataset this reproduces the
    ESMValTool ``multi_datasets.py`` three-panel layout (model | reference on
    a common colorbar, bias below). With several datasets the reference is
    shown once followed by one bias panel per dataset; without a reference a
    plain grid of absolute maps on a common colorbar is drawn.
    """
    try:
        import cartopy.crs as ccrs
    except ImportError:  # pragma: no cover - cartopy ships with esmvaltool
        logger.warning("cartopy is not installed — skipping static map figures")
        return {}

    transform = ccrs.PlateCarree()

    def _projection_for(lats: np.ndarray) -> tuple[ccrs.Projection, tuple | None]:
        """Robinson for global data, polar stereographic for hemispheric-polar.

        Mirrors the ICONEval sea_ice_maps recipes (NorthPolarStereo /
        SouthPolarStereo with central_longitude 270, extent to the data edge).
        """
        if lats.min() >= 20.0:  # noqa: PLR2004
            proj = ccrs.NorthPolarStereo(central_longitude=270)
            return proj, (0, 360, float(lats.min()), 90)
        if lats.max() <= -20.0:  # noqa: PLR2004
            proj = ccrs.SouthPolarStereo(central_longitude=270)
            return proj, (0, 360, -90, float(lats.max()))
        return ccrs.Robinson(central_longitude=10), None

    def _cyclic(pivot: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lons, lats, values), wrapping the seam for global grids."""
        lons = pivot.columns.to_numpy(dtype=float)
        lats = pivot.index.to_numpy(dtype=float)
        vals = pivot.to_numpy(dtype=float)
        if lons.size >= 2:
            spacing = float(np.median(np.diff(lons)))
            if (lons.max() - lons.min()) + 2 * spacing >= 360.0:  # noqa: PLR2004
                vals = np.concatenate([vals, vals[:, :1]], axis=1)
                lons = np.append(lons, lons[0] + 360.0)
        return lons, lats, vals

    def _draw_panel(
        axes: matplotlib.axes.Axes,
        pivot: pd.DataFrame,
        *,
        title: str,
        cmap: str,
        levels: np.ndarray,
        extent: tuple | None,
        title_fontsize: float = _MAP_FONTSIZE,
    ) -> object:
        lons, lats, vals = _cyclic(pivot)
        mappable = axes.contourf(
            lons,
            lats,
            vals,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=transform,
        )
        # multi_datasets.py rasterizes map collections (smaller PDF/SVG output)
        for collection in axes.collections:
            collection.set_rasterized(True)
        axes.coastlines(linewidth=0.5)
        axes.gridlines(**_GRIDLINE_KWARGS)
        if extent is None:
            axes.set_global()
        else:
            axes.set_extent(extent, crs=transform)
        axes.set_title(title, fontsize=title_fontsize)
        return mappable

    def _add_cbar(
        fig: Figure,
        mappable: object,
        axes: object,
        label: str,
        *,
        extra_aspect: float = 0.0,
    ) -> None:
        cbar = fig.colorbar(
            mappable,
            ax=axes,
            orientation=_MAP_CBAR_ORIENTATION,
            aspect=_MAP_CBAR_ASPECT + extra_aspect,
        )
        cbar.set_label(label, fontsize=_MAP_FONTSIZE)
        cbar.ax.tick_params(labelsize=_MAP_FONTSIZE)

    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets_all = _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        long_name, units = _var_label(var_id, variables_df)
        cbar_label = f"{var_id} [{units}]" if units else var_id
        cmap = _mpl_cmap(_colorscale_for(var_id))
        bias_cmap = _mpl_cmap(_bias_colorscale_for(var_id))

        pivots = {
            ds: piv
            for ds in all_datasets
            if (piv := _map_pivot(data_df, ds, var_id)) is not None
        }
        if not pivots:
            continue

        valid_refs = [r for r in ref_datasets_all if r in pivots]
        ref_pivot = _combine_pivots([pivots[r] for r in valid_refs])
        ref_label = valid_refs[0] if len(valid_refs) == 1 else "Multi-obs mean"
        non_ref = [ds for ds in pivots if ds not in valid_refs]
        has_bias = ref_pivot is not None and bool(non_ref)

        vmin, vmax = _data_range(data_df[var_id], var_id)
        levels = _map_levels(vmin, vmax)

        all_lats = np.concatenate(
            [p.index.to_numpy(dtype=float) for p in pivots.values()],
        )
        projection, extent = _projection_for(all_lats)

        if has_bias:
            biases = {
                ds: pivots[ds].subtract(ref_pivot, fill_value=None) for ds in non_ref
            }
            bias_vals = np.concatenate(
                [b.to_numpy(dtype=float).ravel() for b in biases.values()],
            )
            bias_vals = bias_vals[np.isfinite(bias_vals)]
            if bias_vals.size:
                babs = max(
                    abs(float(np.percentile(bias_vals, 2))),
                    abs(float(np.percentile(bias_vals, 98))),
                )
            else:
                babs = 1.0
            babs = babs or 1.0
            bias_levels = _map_levels(-babs, babs)

        if has_bias and len(non_ref) == 1:
            # ESMValTool three-panel layout: model | reference, bias below.
            model = non_ref[0]
            fig = Figure(figsize=_FIGSIZE, layout="constrained")
            gridspec = fig.add_gridspec(
                5,
                4,
                height_ratios=[1.0, 1.0, 0.4, 1.0, 1.0],
            )
            ax_left = fig.add_subplot(gridspec[0:2, 0:2], projection=projection)
            ax_right = fig.add_subplot(gridspec[0:2, 2:4], projection=projection)
            ax_bottom = fig.add_subplot(gridspec[3:5, 1:3], projection=projection)

            m_left = _draw_panel(
                ax_left,
                pivots[model],
                title=str(model),
                cmap=cmap,
                levels=levels,
                extent=extent,
            )
            _draw_panel(
                ax_right,
                ref_pivot,
                title=str(ref_label),
                cmap=cmap,
                levels=levels,
                extent=extent,
            )
            _add_cbar(
                fig,
                m_left,
                [ax_left, ax_right],
                cbar_label,
                extra_aspect=20.0,
            )

            m_bias = _draw_panel(
                ax_bottom,
                biases[model],
                title=f"{model} − {ref_label}",
                cmap=bias_cmap,
                levels=bias_levels,
                extent=extent,
            )
            _add_cbar(fig, m_bias, ax_bottom, f"Δ{cbar_label}")

            _stats_text(
                ax_left, 0.0, 0.95, _fmt_stat(_weighted_mean(pivots[model]), units)
            )
            _stats_text(
                ax_right, 0.0, 0.95, _fmt_stat(_weighted_mean(ref_pivot), units)
            )
            _annotate_bias_panel(
                _stats_text,
                ax_bottom,
                bias=biases[model],
                model_pivot=pivots[model],
                ref_pivot=ref_pivot,
                units=units,
                metrics=_map_metrics_for(metrics_df, var_id, model),
            )
        else:
            # Grid layout: reference first (if any), then bias — or absolute —
            # panels for every dataset, on shared colour levels. With many
            # panels the "model − reference" titles collide, so bias panels
            # carry only the dataset name and the bias reference moves to the
            # figure title.
            if has_bias:
                panels = [(str(ref_label), ref_pivot, cmap, levels, False, None)] + [
                    (str(ds), biases[ds], bias_cmap, bias_levels, True, ds)
                    for ds in non_ref
                ]
            else:
                panels = [
                    (str(ds), pivots[ds], cmap, levels, False, None) for ds in pivots
                ]
            ncols = min(len(panels), 3)
            nrows = (len(panels) + ncols - 1) // ncols
            title_fontsize = _MAP_FONTSIZE if len(panels) <= 6 else 5.0  # noqa: PLR2004
            figsize = _FIGSIZE if len(panels) == 1 else (7.0, 1.0 + 1.9 * nrows)
            fig = Figure(figsize=figsize, layout="constrained")
            groups: dict[bool, tuple[object | None, list]] = {
                False: (None, []),
                True: (None, []),
            }
            for i, (title, pivot, panel_cmap, panel_levels, is_bias, ds) in enumerate(
                panels,
            ):
                axes = fig.add_subplot(nrows, ncols, i + 1, projection=projection)
                mappable = _draw_panel(
                    axes,
                    pivot,
                    title=title,
                    cmap=panel_cmap,
                    levels=panel_levels,
                    extent=extent,
                    title_fontsize=title_fontsize,
                )
                if is_bias and ds is not None:
                    _annotate_bias_panel(
                        _stats_text,
                        axes,
                        bias=pivot,
                        model_pivot=pivots[ds],
                        ref_pivot=ref_pivot,
                        units=units,
                        metrics=_map_metrics_for(metrics_df, var_id, ds),
                    )
                else:
                    _stats_text(
                        axes, 0.0, 0.95, _fmt_stat(_weighted_mean(pivot), units)
                    )
                first, ax_list = groups[is_bias]
                groups[is_bias] = (first if first is not None else mappable, ax_list)
                ax_list.append(axes)
            for is_bias, (mappable, ax_list) in groups.items():
                if mappable is None:
                    continue
                label = f"Δ{cbar_label}" if is_bias else cbar_label
                _add_cbar(fig, mappable, ax_list, label)

        sup = f"{long_name} — bias vs {ref_label}" if has_bias else long_name
        if has_bias and len(non_ref) == 1:
            sup = long_name  # the three-panel bias title already names the reference
        fig.suptitle(sup, fontsize=_MAP_FONTSIZE + 2)
        figures[var_id] = fig

    return figures


# ---------------------------------------------------------------------------
# ZonalLine + ZonalProfile builders
# ---------------------------------------------------------------------------

_PLEV_TICKS = [1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]


def build_static_zonal_line_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Zonal mean vs latitude, multi_datasets.py "variable_vs_lat" style."""
    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets = _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        fig, axes = _new_figure()
        color_idx = 0
        for dataset in all_datasets:
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            if grp.empty:
                continue
            series = grp.groupby("latitude")[var_id].mean().sort_index()
            dtype = grp["data_type"].iloc[0]
            ref_idx = ref_datasets.index(dataset) if dataset in ref_datasets else 0
            axes.plot(
                series.index,
                series.values,
                label=None if dtype == "other" else str(dataset),
                **_line_kwargs(dtype, color_idx, ref_idx),
            )
            if dtype == "to_benchmark":
                color_idx += 1

        axes.set_xlabel("latitude [°N]")
        ylabel = f"{var_id} [{units}]" if units else var_id
        _finish_line_axes(axes, title=long_name, ylabel=ylabel)
        figures[var_id] = fig

    return figures


def build_static_zonal_profile_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Zonal mean profiles (lat × pressure), "zonal_mean_profile" style.

    Same layout rules as the static maps: three panels for a single
    benchmarked dataset with a reference, otherwise a reference + bias (or
    absolute) grid. Log-pressure axis decreasing upward, vertical colorbars,
    stats text at the multi_datasets.py positions (0.01 / 0.7).
    """
    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets_all = _get_ref_datasets(data_df, metrics_df)

    def _profile_pivot(dataset: str, var_id: str) -> pd.DataFrame | None:
        sub = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
        if sub.empty:
            return None
        p = sub.pivot_table(
            index="air_pressure",
            columns="latitude",
            values=var_id,
            aggfunc="mean",
        ).sort_index()
        return p if not p.empty else None

    def _draw_profile(
        axes: matplotlib.axes.Axes,
        pivot: pd.DataFrame,
        *,
        title: str,
        cmap: str,
        levels: np.ndarray,
    ) -> object:
        mappable = axes.contourf(
            pivot.columns.to_numpy(dtype=float),
            pivot.index.to_numpy(dtype=float),
            pivot.to_numpy(dtype=float),
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        for collection in axes.collections:
            collection.set_rasterized(True)
        axes.set_yscale("log")
        axes.invert_yaxis()
        pmin = float(pivot.index.min())
        pmax = float(pivot.index.max())
        ticks = [t for t in _PLEV_TICKS if pmin <= t <= pmax]
        axes.set_yticks(ticks)
        axes.set_yticklabels([str(t) for t in ticks], fontsize=_MAP_FONTSIZE)
        axes.minorticks_off()  # y_minor_formatter: '' in multi_datasets.py
        axes.tick_params(labelsize=_MAP_FONTSIZE)
        axes.set_title(title, fontsize=_MAP_FONTSIZE)
        return mappable

    def _add_cbar(
        fig: Figure,
        mappable: object,
        axes: object,
        label: str,
    ) -> None:
        cbar = fig.colorbar(mappable, ax=axes, orientation="vertical")
        cbar.set_label(label, fontsize=_MAP_FONTSIZE)
        cbar.ax.tick_params(labelsize=_MAP_FONTSIZE)

    for var_id in get_var_cols(data_df):
        long_name, units = _var_label(var_id, variables_df)
        cbar_label = f"{var_id} [{units}]" if units else var_id
        cmap = _mpl_cmap(_colorscale_for(var_id))
        bias_cmap = _mpl_cmap(_bias_colorscale_for(var_id))

        pivots = {
            ds: piv
            for ds in all_datasets
            if (piv := _profile_pivot(ds, var_id)) is not None
        }
        if not pivots:
            continue

        valid_refs = [r for r in ref_datasets_all if r in pivots]
        ref_pivot = _combine_pivots([pivots[r] for r in valid_refs])
        ref_label = valid_refs[0] if len(valid_refs) == 1 else "Multi-obs mean"
        non_ref = [ds for ds in pivots if ds not in valid_refs]
        has_bias = ref_pivot is not None and bool(non_ref)
        # mypy cannot narrow ref_pivot through the has_bias flag
        ref_pivot_t = None if ref_pivot is None else ref_pivot.T

        vmin, vmax = _data_range(data_df[var_id], var_id)
        levels = _map_levels(vmin, vmax)

        if has_bias:
            biases = {ds: pivots[ds].subtract(ref_pivot) for ds in non_ref}
            bias_vals = np.concatenate(
                [b.to_numpy(dtype=float).ravel() for b in biases.values()],
            )
            bias_vals = bias_vals[np.isfinite(bias_vals)]
            if bias_vals.size:
                babs = max(
                    abs(float(np.percentile(bias_vals, 2))),
                    abs(float(np.percentile(bias_vals, 98))),
                )
            else:
                babs = 1.0
            babs = babs or 1.0
            bias_levels = _map_levels(-babs, babs)

        if has_bias and len(non_ref) == 1:
            model = non_ref[0]
            fig = Figure(figsize=_FIGSIZE, layout="constrained")
            gridspec = fig.add_gridspec(
                5,
                4,
                height_ratios=[1.0, 1.0, 0.4, 1.0, 1.0],
            )
            ax_left = fig.add_subplot(gridspec[0:2, 0:2])
            ax_right = fig.add_subplot(gridspec[0:2, 2:4])
            ax_bottom = fig.add_subplot(gridspec[3:5, 1:3])

            m_left = _draw_profile(
                ax_left, pivots[model], title=str(model), cmap=cmap, levels=levels
            )
            _draw_profile(
                ax_right, ref_pivot, title=str(ref_label), cmap=cmap, levels=levels
            )
            _add_cbar(fig, m_left, [ax_left, ax_right], cbar_label)
            m_bias = _draw_profile(
                ax_bottom,
                biases[model],
                title=f"{model} − {ref_label}",
                cmap=bias_cmap,
                levels=bias_levels,
            )
            _add_cbar(fig, m_bias, ax_bottom, f"Δ{cbar_label}")

            ax_right.set_yticklabels([])
            for axes in (ax_left, ax_bottom):
                axes.set_ylabel("Pressure (hPa)", fontsize=_MAP_FONTSIZE)
            for axes in (ax_left, ax_right, ax_bottom):
                axes.set_xlabel("latitude [°N]", fontsize=_MAP_FONTSIZE)

            _stats_text(
                ax_left, 0.01, 0.95, _fmt_stat(_weighted_mean(pivots[model].T), units)
            )
            _stats_text(
                ax_right, 0.01, 0.95, _fmt_stat(_weighted_mean(ref_pivot_t), units)
            )
            _annotate_bias_panel(
                _stats_text,
                ax_bottom,
                bias=biases[model].T,
                model_pivot=pivots[model].T,
                ref_pivot=ref_pivot_t,
                units=units,
                metrics=_map_metrics_for(metrics_df, var_id, model),
                x_avg=0.01,
                x_bias=0.7,
            )
        else:
            if ref_pivot is not None and non_ref:  # has_bias, mypy-narrowed
                panels = [(str(ref_label), ref_pivot, cmap, levels, False, None)] + [
                    (str(ds), biases[ds], bias_cmap, bias_levels, True, ds)
                    for ds in non_ref
                ]
            else:
                panels = [
                    (str(ds), pivots[ds], cmap, levels, False, None) for ds in pivots
                ]
            ncols = min(len(panels), 3)
            nrows = (len(panels) + ncols - 1) // ncols
            figsize = _FIGSIZE if len(panels) == 1 else (7.0, 1.0 + 2.2 * nrows)
            fig = Figure(figsize=figsize, layout="constrained")
            groups: dict[bool, tuple[object | None, list]] = {
                False: (None, []),
                True: (None, []),
            }
            for i, (title, pivot, panel_cmap, panel_levels, is_bias, ds) in enumerate(
                panels,
            ):
                axes = fig.add_subplot(nrows, ncols, i + 1)
                mappable = _draw_profile(
                    axes, pivot, title=title, cmap=panel_cmap, levels=panel_levels
                )
                if i % ncols == 0:
                    axes.set_ylabel("Pressure (hPa)", fontsize=_MAP_FONTSIZE)
                else:
                    axes.set_yticklabels([])
                if i // ncols == nrows - 1:
                    axes.set_xlabel("latitude [°N]", fontsize=_MAP_FONTSIZE)
                if is_bias and ds is not None:
                    _annotate_bias_panel(
                        _stats_text,
                        axes,
                        bias=pivot.T,
                        model_pivot=pivots[ds].T,
                        ref_pivot=ref_pivot_t,
                        units=units,
                        metrics=_map_metrics_for(metrics_df, var_id, ds),
                        x_avg=0.01,
                        x_bias=0.7,
                    )
                else:
                    _stats_text(
                        axes, 0.01, 0.95, _fmt_stat(_weighted_mean(pivot.T), units)
                    )
                first, ax_list = groups[is_bias]
                groups[is_bias] = (first if first is not None else mappable, ax_list)
                ax_list.append(axes)
            for is_bias, (mappable, ax_list) in groups.items():
                if mappable is None:
                    continue
                label = f"Δ{cbar_label}" if is_bias else cbar_label
                _add_cbar(fig, mappable, ax_list, label)

        sup = f"{long_name} — bias vs {ref_label}" if has_bias else long_name
        if has_bias and len(non_ref) == 1:
            sup = long_name
        fig.suptitle(sup, fontsize=_MAP_FONTSIZE + 2)
        figures[var_id] = fig

    return figures


# ---------------------------------------------------------------------------
# Registry + export entry point
# ---------------------------------------------------------------------------

# Keyed by the detect_diag_type() strings, like _FIGURE_BUILDERS in
# _figures.py. Types without a static builder yet are skipped (with a log
# message) by generate_static_plots().
_STATIC_FIGURE_BUILDERS: dict[str, Callable] = {
    "Map": build_static_map_figures,
    "TimeSeries": build_static_timeseries_figures,
    "ZonalLine": build_static_zonal_line_figures,
    "ZonalProfile": build_static_zonal_profile_figures,
}


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _save_figure(
    fig: Figure,
    base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    """Write one file per format next to *base* (suffix added per format)."""
    written: list[Path] = []
    for fmt in formats:
        out_path = base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, format=fmt, **_SAVEFIG_KWARGS)
        written.append(out_path)
    return written


def generate_static_plots(
    db_paths: list[Path],
    out_dir: Path,
    *,
    formats: Iterable[str] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """Render static figures from result databases into *out_dir*.

    One file per (database, diagnostic, variable, format) is written to
    ``out_dir/<db_stem>/<diagnostic>/<var_id>.<format>``. Diagnostic types
    without a static builder yet are skipped with a log message.

    Args:
        db_paths: Paths to one or more DuckDB .ddb files.
        out_dir: Output directory (created if missing).
        formats: Image formats understood by matplotlib (e.g. "png", "pdf").
        dpi: Resolution for raster formats (ESMValTool default: 300).

    Returns
    -------
        The list of written file paths.
    """
    written: list[Path] = []
    for db_path in db_paths:
        db_data = read_database(db_path)
        if not db_data:
            logger.warning("No diagnostic databases found in {}", db_path)
            continue
        for diag_name, tables in db_data.items():
            data_df = tables.get("raw_output", pd.DataFrame())
            metrics_df = tables.get("metrics", pd.DataFrame())
            variables_df = tables.get("variables", pd.DataFrame())
            if data_df.empty:
                continue
            diag_type = detect_diag_type(data_df.columns.tolist())
            builder = _STATIC_FIGURE_BUILDERS.get(diag_type)
            if builder is None:
                logger.info(
                    "Skipping {} ({}): no static builder for this type yet",
                    diag_name,
                    diag_type,
                )
                continue
            figures = builder(data_df, variables_df, metrics_df)
            diag_dir = (
                out_dir / _safe_filename(db_path.stem) / _safe_filename(diag_name)
            )
            diag_dir.mkdir(parents=True, exist_ok=True)
            for var_id, fig in figures.items():
                base = diag_dir / _safe_filename(var_id)
                written.extend(_save_figure(fig, base, formats, dpi))
    logger.info("Wrote {} static figure file(s) to {}", len(written), out_dir)
    return written
