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

import warnings
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
    _weighted_mean,
)
from climateeval.report._db import detect_diag_type, get_var_cols, read_database
from climateeval.report._style import (
    _OTHER_COLORS,
    _REFERENCE_COLORS,
    _SEABORN_COLORS,
    _bias_colorscale_for,
    _colorscale_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
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

# Line styling per data_type, the same scheme as the interactive report
# (_figures._get_line_properties): the benchmarked model drawn on top as a
# solid line in its palette colour, references solid in dark colours (black
# first), compared datasets thin with a distinct colour + dash pair each.
_BENCHMARK_KWARGS = {"linewidth": 1.25, "linestyle": "solid", "zorder": 2.5}
_REFERENCE_KWARGS = {"linewidth": 1.0, "linestyle": "solid", "zorder": 2.4}
_OTHER_KWARGS = {"linewidth": 0.75, "zorder": 1.0}
_OTHER_LINESTYLES = ["dashed", "dotted", "dashdot", (0, (6, 2)), (0, (6, 2, 1, 2))]

_SEABORN_STYLE = "ticks"  # multi_datasets.py seaborn_settings default


def _new_figure() -> tuple[Figure, matplotlib.axes.Axes]:
    """Return an ICONEval-styled (figure, axes) pair (no pyplot state)."""
    with sns.axes_style(_SEABORN_STYLE):
        fig = Figure(figsize=_FIGSIZE)
        axes = fig.add_subplot(111)
    return fig, axes


def _line_kwargs(dtype: str, idx: int) -> dict:
    """Return matplotlib plot kwargs for the ``idx``-th dataset of a type."""
    if dtype == "reference":
        return {
            **_REFERENCE_KWARGS,
            "color": _REFERENCE_COLORS[idx % len(_REFERENCE_COLORS)],
        }
    if dtype == "to_benchmark":
        return {
            **_BENCHMARK_KWARGS,
            "color": _SEABORN_COLORS[idx % len(_SEABORN_COLORS)],
        }
    return {
        **_OTHER_KWARGS,
        "color": _OTHER_COLORS[idx % len(_OTHER_COLORS)],
        "linestyle": _OTHER_LINESTYLES[idx % len(_OTHER_LINESTYLES)],
    }


def _line_styles(data_df: pd.DataFrame, datasets: Iterable) -> dict:
    """Line kwargs per dataset, numbered within each data type.

    Assigned once per figure so a dataset keeps its look in every panel.
    """
    dtypes = data_df.drop_duplicates("data_id").set_index("data_id")["data_type"]
    counts: dict[str, int] = {}
    styles = {}
    for dataset in datasets:
        dtype = str(dtypes.get(dataset, "other"))
        styles[dataset] = _line_kwargs(dtype, counts.get(dtype, 0))
        counts[dtype] = counts.get(dtype, 0) + 1
    return styles


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
    _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        fig, axes = _new_figure()
        _plot_reasonable_range(axes, variables_df, var_id)

        # Use markers when there are too few points for a visible line.
        marker = "o" if data_df["time"].nunique() < 3 else None

        styles = _line_styles(data_df, all_datasets)
        for dataset in all_datasets:
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            sub = grp[["time", var_id]].sort_values("time")
            if sub.empty:
                continue
            axes.plot(
                sub["time"],
                sub[var_id],
                label=str(dataset),
                marker=marker,
                **styles[dataset],
            )

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


def _labeller(display_names: Mapping[str, str] | None) -> Callable[[object], str]:
    """Return a function mapping a dataset id to its displayed name.

    Presentation only.  Nothing downstream of the title string sees it, so the
    scoring, the metrics and the output paths keep the real identifier.
    """
    names = dict(display_names or {})
    return lambda ds: names.get(str(ds), str(ds))


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


def _draw_map_panel(  # noqa: PLR0913
    axes: matplotlib.axes.Axes,
    pivot: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    levels: np.ndarray,
    extent: tuple | None,
    transform: object,
    title_fontsize: float = _MAP_FONTSIZE,
) -> object:
    """Draw filled contours with coastlines, the one map panel both builders use."""
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
    display_names: Mapping[str, str] | None = None,
) -> dict[str, Figure]:
    """Global maps on a Robinson projection, ESMValTool monitor style.

    With a reference and exactly one benchmarked dataset this reproduces the
    ESMValTool ``multi_datasets.py`` three-panel layout (model | reference on
    a common colorbar, bias below). With several datasets the reference is
    shown once followed by one bias panel per dataset; without a reference a
    plain grid of absolute maps on a common colorbar is drawn.

    ``display_names`` maps a dataset id to the name shown in panel titles.  It
    is presentation only: the ids in the data, the metrics and the output file
    names are untouched, so a figure can say "ESM" while the record still says
    what the run actually was.
    """
    display = _labeller(display_names)
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
        return _draw_map_panel(
            axes,
            pivot,
            title=title,
            cmap=cmap,
            levels=levels,
            extent=extent,
            transform=transform,
            title_fontsize=title_fontsize,
        )

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
            # Model | observations | bias, left to right in one row (user
            # request 2026-09-24).  The first two share a colour scale so the
            # eye can compare them; the third is diverging about zero.
            model = non_ref[0]
            fig = Figure(figsize=(12.0, 3.2), layout="constrained")
            ax_left = fig.add_subplot(1, 3, 1, projection=projection)
            ax_right = fig.add_subplot(1, 3, 2, projection=projection)
            ax_bottom = fig.add_subplot(1, 3, 3, projection=projection)

            m_left = _draw_panel(
                ax_left,
                pivots[model],
                title=display(model),
                cmap=cmap,
                levels=levels,
                extent=extent,
            )
            _draw_panel(
                ax_right,
                ref_pivot,
                title=display(ref_label),
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
                title=f"{display(model)} − {display(ref_label)}",
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
                panels = [
                    (display(ref_label), ref_pivot, cmap, levels, False, None)
                ] + [
                    (display(ds), biases[ds], bias_cmap, bias_levels, True, ds)
                    for ds in non_ref
                ]
            else:
                panels = [
                    (display(ds), pivots[ds], cmap, levels, False, None)
                    for ds in pivots
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

        sup = f"{long_name} — bias vs {display(ref_label)}" if has_bias else long_name
        if has_bias and len(non_ref) == 1:
            sup = long_name  # the three-panel bias title already names the reference
        fig.suptitle(sup, fontsize=_MAP_FONTSIZE + 2)
        figures[var_id] = fig

    return figures


def build_static_map_sheet(  # noqa: PLR0913, PLR0915
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    var_ids: Sequence[str],
    *,
    metrics_df: pd.DataFrame | None = None,
    display_names: Mapping[str, str] | None = None,
    panel_width: float = 4.2,
    panel_height: float = 2.0,
) -> Figure | None:
    """One sheet: a row per variable, columns model | observations | bias.

    Built for figures that go into a paper rather than into the report, so the
    column headings are drawn once at the top instead of on every row, and the
    row label carries the variable name.  Each middle panel still names its own
    reference dataset, because the references differ per row.

    Returns None when no requested variable has both a model and a reference.
    """
    try:
        import cartopy.crs as ccrs
    except ImportError:  # pragma: no cover - cartopy ships with esmvaltool
        logger.warning("cartopy is not installed — skipping the map sheet")
        return None

    display = _labeller(display_names)
    transform = ccrs.PlateCarree()
    projection = ccrs.Robinson(central_longitude=10)

    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets_all = _get_ref_datasets(data_df, metrics_df)

    rows = []
    for var_id in var_ids:
        if var_id not in data_df.columns:
            logger.warning("Skipping {} in the sheet: not in this database", var_id)
            continue
        pivots = {
            ds: piv
            for ds in all_datasets
            if (piv := _map_pivot(data_df, ds, var_id)) is not None
        }
        valid_refs = [r for r in ref_datasets_all if r in pivots]
        ref_pivot = _combine_pivots([pivots[r] for r in valid_refs])
        non_ref = [ds for ds in pivots if ds not in valid_refs]
        if ref_pivot is None or len(non_ref) != 1:
            logger.warning(
                "Skipping {} in the sheet: needs one model and a reference", var_id
            )
            continue
        rows.append((var_id, non_ref[0], pivots, ref_pivot, valid_refs))

    if not rows:
        return None

    # The column headings are drawn once, so they can only name one model.
    # Rows that resolved to different models would be silently mislabelled.
    models = {row[1] for row in rows}
    if len(models) > 1:
        msg = (
            "build_static_map_sheet needs one model across all rows; the "
            f"requested variables resolve to {sorted(map(str, models))}"
        )
        raise ValueError(msg)

    fig = Figure(
        figsize=(3 * panel_width, len(rows) * panel_height + 0.6),
        layout="constrained",
    )
    grid = fig.add_gridspec(len(rows), 3)

    for r, (var_id, model, pivots, ref_pivot, valid_refs) in enumerate(rows):
        long_name, units = _var_label(var_id, variables_df)
        cbar_label = f"{var_id} [{units}]" if units else var_id
        cmap = _mpl_cmap(_colorscale_for(var_id))
        bias_cmap = _mpl_cmap(_bias_colorscale_for(var_id))
        ref_name = valid_refs[0] if len(valid_refs) == 1 else "Multi-obs mean"

        vmin, vmax = _data_range(data_df[var_id], var_id)
        levels = _map_levels(vmin, vmax)
        bias = pivots[model].subtract(ref_pivot, fill_value=None)
        with warnings.catch_warnings():
            # An all-NaN bias is a legitimate degenerate case here, and numpy's
            # "All-NaN slice" warning on stderr is noise in a figure pipeline.
            warnings.simplefilter("ignore", RuntimeWarning)
            babs = float(np.nanpercentile(np.abs(bias.to_numpy(dtype=float)), 98))
        # Catches NaN (all-NaN bias), inf, and an exactly-zero bias, all of
        # which would otherwise give a degenerate colour scale.
        if not np.isfinite(babs) or babs == 0:
            babs = 1.0
        bias_levels = _map_levels(-babs, babs)

        axes = [fig.add_subplot(grid[r, c], projection=projection) for c in range(3)]
        m_data = _draw_map_panel(
            axes[0],
            pivots[model],
            title="",
            cmap=cmap,
            levels=levels,
            extent=None,
            transform=transform,
        )
        _draw_map_panel(
            axes[1],
            ref_pivot,
            title=display(ref_name),
            cmap=cmap,
            levels=levels,
            extent=None,
            transform=transform,
        )
        m_bias = _draw_map_panel(
            axes[2],
            bias,
            title="",
            cmap=bias_cmap,
            levels=bias_levels,
            extent=None,
            transform=transform,
        )

        cb_data = fig.colorbar(
            m_data, ax=axes[:2], orientation="vertical", aspect=12, pad=0.01
        )
        cb_data.set_label(cbar_label, fontsize=_MAP_FONTSIZE)
        cb_data.ax.tick_params(labelsize=_MAP_FONTSIZE)
        cb_bias = fig.colorbar(
            m_bias, ax=axes[2], orientation="vertical", aspect=12, pad=0.01
        )
        cb_bias.set_label(f"\u0394{cbar_label}", fontsize=_MAP_FONTSIZE)
        cb_bias.ax.tick_params(labelsize=_MAP_FONTSIZE)

        _stats_text(axes[0], 0.0, 0.95, _fmt_stat(_weighted_mean(pivots[model]), units))
        _stats_text(axes[1], 0.0, 0.95, _fmt_stat(_weighted_mean(ref_pivot), units))
        _annotate_bias_panel(
            _stats_text,
            axes[2],
            bias=bias,
            model_pivot=pivots[model],
            ref_pivot=ref_pivot,
            units=units,
            metrics=_map_metrics_for(metrics_df, var_id, model),
        )
        axes[0].text(
            -0.04,
            0.5,
            long_name,
            transform=axes[0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=_MAP_FONTSIZE + 1,
        )
        if r == 0:
            headings = (
                display(model),
                "Observations",
                f"{display(model)} \u2212 observations",
            )
            for axis, heading in zip(axes, headings, strict=True):
                axis.set_title(
                    heading
                    if axis is not axes[1]
                    else f"{heading}\n{display(ref_name)}",
                    fontsize=_MAP_FONTSIZE + 2,
                    fontweight="bold",
                )

    return fig


# ---------------------------------------------------------------------------
# AnnualCycle / DiurnalCycle / Histogram builders
# ---------------------------------------------------------------------------


def _build_static_cycle_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None,
    *,
    coord: str,
    xlabel: str,
    xticks: list[int],
    xticklabels: list[str] | None = None,
) -> dict[str, Figure]:
    """Shared 1D-cycle line plot (multi_datasets.py annual/diurnal cycle)."""
    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        fig, axes = _new_figure()
        styles = _line_styles(data_df, all_datasets)
        for dataset in all_datasets:
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            if grp.empty:
                continue
            series = grp.groupby(coord)[var_id].mean().sort_index()
            axes.plot(
                series.index,
                series.values,
                label=str(dataset),
                **styles[dataset],
            )

        axes.set_xticks(xticks)
        if xticklabels is not None:
            axes.set_xticklabels(xticklabels)
        axes.set_xlabel(xlabel)
        ylabel = f"{var_id} [{units}]" if units else var_id
        _finish_line_axes(axes, title=long_name, ylabel=ylabel)
        figures[var_id] = fig

    return figures


def build_static_annual_cycle_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Annual cycle (Jan–Dec), multi_datasets.py "annual_cycle" style."""
    return _build_static_cycle_figures(
        data_df,
        variables_df,
        metrics_df,
        coord="month_number",
        xlabel="month",
        xticks=list(range(1, 13)),
        xticklabels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
    )


def build_static_diurnal_cycle_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Diurnal cycle (0–24 h), multi_datasets.py "diurnal_cycle" style."""
    return _build_static_cycle_figures(
        data_df,
        variables_df,
        metrics_df,
        coord="hour",
        xlabel="hour",
        xticks=list(range(0, 25, 3)),
    )


def build_static_histogram_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """Histograms, ESMValTool plot_histogram.py style.

    Overlapping semi-transparent bars per dataset at the stored bin centres,
    "Density" on y (log for precipitation, like the ICONEval histogram
    recipes), reference black / benchmarked coloured.
    """
    from climateeval.report._db import get_density_cols
    from climateeval.report._style import _base_var

    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())

    for density_col in get_density_cols(data_df):
        var_id = density_col.replace("density_", "")
        sub = data_df[["data_id", "data_type", var_id, density_col]].dropna(
            subset=[density_col],
        )
        if sub.empty:
            continue
        long_name, units = _var_label(var_id, variables_df)
        log_y = _base_var(var_id) == "pr"

        fig, axes = _new_figure()
        styles = _line_styles(data_df, all_datasets)
        for dataset in all_datasets:
            ds_sub = sub[sub["data_id"] == dataset].sort_values(var_id)
            if ds_sub.empty:
                continue
            kwargs = styles[dataset]
            centers = ds_sub[var_id].to_numpy(dtype=float)
            heights = ds_sub[density_col].to_numpy(dtype=float)
            if log_y:
                heights = np.where(heights > 0, heights, 0.0)
            if centers.size > 1:
                widths = np.diff(centers, prepend=2 * centers[0] - centers[1])
            else:
                widths = np.array([1.0])
            axes.bar(
                centers,
                heights,
                width=widths,
                alpha=0.5,
                color=kwargs["color"],
                label=str(dataset),
                zorder=kwargs["zorder"],
            )

        if log_y:
            axes.set_yscale("log")
        xlabel = f"{long_name} [{units}]" if units else long_name
        axes.set_xlabel(xlabel)
        _finish_line_axes(axes, title=long_name, ylabel="Density")
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
    _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        fig, axes = _new_figure()
        styles = _line_styles(data_df, all_datasets)
        for dataset in all_datasets:
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            if grp.empty:
                continue
            series = grp.groupby("latitude")[var_id].mean().sort_index()
            axes.plot(
                series.index,
                series.values,
                label=str(dataset),
                **styles[dataset],
            )

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
        if p.empty:
            return None
        if p.index.max() > 2000.0:  # noqa: PLR2004 - stored in Pa
            p.index = p.index / 100.0  # Pa → hPa
        return p

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
# Scalar builder
# ---------------------------------------------------------------------------

_SCALAR_TYPE_COLORS = {
    "to_benchmark": "#4C72B0",
    "reference": "#c0860a",
    "other": "#aaaaaa",
}


def build_static_bar_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,  # noqa: ARG001
) -> dict[str, Figure]:
    """Horizontal bar chart for scalar (0-D) diagnostics, one per variable.

    Mirrors the interactive Scalar figures: benchmarked dataset on top in
    blue, references in gold, auxiliary datasets in light gray, values
    annotated next to the bars.
    """
    figures: dict[str, Figure] = {}
    type_order = {"to_benchmark": 0, "reference": 1, "other": 2}

    for var_id in get_var_cols(data_df):
        sub = data_df[["data_id", "data_type", var_id]].dropna(subset=[var_id]).copy()
        if sub.empty:
            continue
        long_name, units = _var_label(var_id, variables_df)

        sub["_type_rank"] = sub["data_type"].map(type_order).fillna(3)
        sub = sub.sort_values(["_type_rank", var_id], ascending=[True, True])

        n = len(sub)
        with sns.axes_style(_SEABORN_STYLE):
            fig = Figure(
                figsize=(7.0, max(2.5, 0.35 * n + 1.5)),
                layout="constrained",
            )
            axes = fig.add_subplot(111)

        # Plot bottom-up so to_benchmark ends on top
        y_pos = range(n - 1, -1, -1)
        values = sub[var_id].to_numpy(dtype=float)
        colors = [_SCALAR_TYPE_COLORS.get(str(t), "#aaaaaa") for t in sub["data_type"]]
        axes.barh(list(y_pos), values, color=colors, height=0.7)
        axes.set_yticks(list(y_pos))
        axes.set_yticklabels(
            sub["data_id"].tolist(),
            fontsize=_MAP_FONTSIZE,
        )
        for y, v in zip(y_pos, values, strict=True):
            axes.annotate(
                f"{v:.3g}",
                xy=(v, y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=_MAP_STATS_FONTSIZE,
            )
        axes.grid(visible=True, axis="x", **_GRIDLINE_KWARGS)
        axes.set_axisbelow(True)
        axes.margins(x=0.15)
        axes.set_xlabel(f"{var_id} [{units}]" if units else var_id)
        axes.set_title(long_name)
        figures[var_id] = fig

    return figures


# ---------------------------------------------------------------------------
# Hovmöller + QBO builders
# ---------------------------------------------------------------------------


def build_static_hovmoller_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,  # noqa: ARG001
) -> dict[str, Figure]:
    """Hovmöller diagrams: longitude × time anomalies, one panel per dataset.

    Stacked pcolormesh panels on a shared symmetric colour range (single
    colorbar), time increasing downward like the interactive report.
    """
    from climateeval.report._style import _hovmoller_colorscale_for

    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)
        cmap = _mpl_cmap(_hovmoller_colorscale_for(var_id))

        pivots: dict[str, pd.DataFrame] = {}
        for ds in all_datasets:
            sub = data_df[data_df["data_id"] == ds].dropna(subset=[var_id])
            if sub.empty:
                continue
            p = sub.pivot_table(
                index="time",
                columns="longitude",
                values=var_id,
                aggfunc="mean",
            )
            p = p.dropna(how="all").dropna(axis=1, how="all")
            if not p.empty:
                pivots[ds] = p
        if not pivots:
            continue

        all_vals = np.concatenate(
            [p.to_numpy(dtype=float).ravel() for p in pivots.values()],
        )
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size:
            vabs = max(
                abs(float(np.percentile(all_vals, 2))),
                abs(float(np.percentile(all_vals, 98))),
            )
        else:
            vabs = 1.0
        vabs = vabs or 1.0

        nrows = len(pivots)
        fig = Figure(figsize=(7.0, 2.0 + 2.2 * nrows), layout="constrained")
        axes_list = []
        mappable = None
        for i, (ds, pivot) in enumerate(pivots.items()):
            axes = fig.add_subplot(nrows, 1, i + 1)
            mappable = axes.pcolormesh(
                pivot.columns.to_numpy(dtype=float),
                pivot.index.to_numpy(),
                pivot.to_numpy(dtype=float),
                cmap=cmap,
                vmin=-vabs,
                vmax=vabs,
                rasterized=True,
            )
            axes.invert_yaxis()
            axes.set_title(str(ds), fontsize=_MAP_FONTSIZE)
            axes.set_ylabel("Time", fontsize=_MAP_FONTSIZE)
            axes.tick_params(labelsize=_MAP_FONTSIZE)
            if i == nrows - 1:
                axes.set_xlabel("Longitude", fontsize=_MAP_FONTSIZE)
            axes_list.append(axes)
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axes_list, orientation="vertical")
            cbar.set_label(
                f"{var_id} [{units}]" if units else var_id,
                fontsize=_MAP_FONTSIZE,
            )
            cbar.ax.tick_params(labelsize=_MAP_FONTSIZE)

        fig.suptitle(long_name, fontsize=_MAP_FONTSIZE + 2)
        figures[var_id] = fig

    return figures


def build_static_qbo_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Figure]:
    """QBO figures mirroring the interactive layout.

    Left: stacked time × pressure heatmaps (one per dataset, shared symmetric
    colour range). Right: amplitude profile (std vs pressure). Bottom: QBO
    index (wind at ~50 hPa vs time).
    """
    figures: dict[str, Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    _get_ref_datasets(data_df, metrics_df)
    qbo_ticks = [100, 70, 50, 30, 20, 10, 5, 1]
    index_plev_pa = 5000.0  # ~50 hPa

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)
        cmap = _mpl_cmap(_colorscale_for(var_id))

        pivots: dict[str, pd.DataFrame] = {}
        for ds in all_datasets:
            sub = data_df[data_df["data_id"] == ds].dropna(subset=[var_id])
            if sub.empty:
                continue
            p = sub.pivot_table(
                index="air_pressure",
                columns="time",
                values=var_id,
                aggfunc="mean",
            )
            p = p.dropna(how="all").dropna(axis=1, how="all")
            if not p.empty:
                p.index = p.index / 100.0  # Pa → hPa
                pivots[ds] = p
        if not pivots:
            continue

        arr = data_df[var_id].dropna().to_numpy(dtype=float)
        vabs = max(
            abs(float(np.percentile(arr, 2))),
            abs(float(np.percentile(arr, 98))),
        )
        vabs = vabs or 1.0

        n_ds = len(pivots)
        fig = Figure(figsize=(7.0, 1.8 * n_ds + 2.6), layout="constrained")
        gridspec = fig.add_gridspec(
            n_ds + 1,
            5,
            height_ratios=[1.8] * n_ds + [1.6],
        )

        # Stacked heatmaps (left 4/5 columns)
        heatmap_axes = []
        mappable = None
        for i, (ds, pivot) in enumerate(pivots.items()):
            axes = fig.add_subplot(gridspec[i, 0:4])
            mappable = axes.pcolormesh(
                pivot.columns.to_numpy(),
                pivot.index.to_numpy(dtype=float),
                pivot.to_numpy(dtype=float),
                cmap=cmap,
                vmin=-vabs,
                vmax=vabs,
                rasterized=True,
            )
            axes.set_yscale("log")
            axes.set_ylim(float(pivot.index.max()), float(pivot.index.min()))
            ticks = [
                t for t in qbo_ticks if pivot.index.min() <= t <= pivot.index.max()
            ]
            axes.set_yticks(ticks)
            axes.set_yticklabels([str(t) for t in ticks], fontsize=_MAP_FONTSIZE)
            axes.minorticks_off()
            axes.set_title(str(ds), fontsize=_MAP_FONTSIZE)
            axes.set_ylabel("Pressure (hPa)", fontsize=_MAP_FONTSIZE)
            axes.tick_params(labelsize=_MAP_FONTSIZE)
            if i < n_ds - 1:
                axes.set_xticklabels([])
            heatmap_axes.append(axes)
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=heatmap_axes, orientation="vertical")
            cbar.set_label(
                f"{var_id} [{units}]" if units else var_id,
                fontsize=_MAP_FONTSIZE,
            )
            cbar.ax.tick_params(labelsize=_MAP_FONTSIZE)

        # Amplitude profile (right column, spans the heatmap rows)
        ax_amp = fig.add_subplot(gridspec[0:n_ds, 4])
        styles = _line_styles(data_df, list(pivots))
        for ds in pivots:
            grp = data_df[data_df["data_id"] == ds].dropna(subset=[var_id])
            std_profile = (
                grp.assign(plev_hpa=grp["air_pressure"] / 100.0)
                .groupby("plev_hpa")[var_id]
                .std()
                .sort_index()
            )
            if std_profile.empty:
                continue
            ax_amp.plot(
                std_profile.values,
                std_profile.index,
                **styles[ds],
            )
        ax_amp.set_yscale("log")
        ax_amp.invert_yaxis()
        ax_amp.minorticks_off()
        ax_amp.set_yticks([])
        ax_amp.set_title("Amplitude", fontsize=_MAP_FONTSIZE)
        ax_amp.set_xlabel(f"σ [{units}]", fontsize=_MAP_FONTSIZE)
        ax_amp.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_amp.tick_params(labelsize=_MAP_FONTSIZE)
        ax_amp.tick_params(axis="x", labelrotation=45)

        # QBO index (bottom, full width)
        ax_idx = fig.add_subplot(gridspec[n_ds, :])
        for ds in pivots:
            grp = data_df[data_df["data_id"] == ds].dropna(subset=[var_id])
            at_50 = grp[np.isclose(grp["air_pressure"], index_plev_pa)]
            if at_50.empty:
                continue
            series = at_50.groupby("time")[var_id].mean().sort_index()
            ax_idx.plot(
                series.index,
                series.values,
                label=str(ds),
                **styles[ds],
            )
        ax_idx.grid(visible=True, **_GRIDLINE_KWARGS)
        ax_idx.set_axisbelow(True)
        ax_idx.set_title("QBO index (~50 hPa)", fontsize=_MAP_FONTSIZE)
        ax_idx.set_ylabel(
            f"{var_id} [{units}]" if units else var_id,
            fontsize=_MAP_FONTSIZE,
        )
        ax_idx.tick_params(labelsize=_MAP_FONTSIZE)
        handles, labels = ax_idx.get_legend_handles_labels()
        if handles:
            ax_idx.legend(
                handles,
                labels,
                fontsize=_MAP_FONTSIZE,
                **_LEGEND_KWARGS,
                bbox_to_anchor=(0.5, -0.25),
                ncols=min(len(handles), 3),
            )

        fig.suptitle(long_name, fontsize=_MAP_FONTSIZE + 2)
        figures[var_id] = fig

    return figures


# ---------------------------------------------------------------------------
# Portrait plot (overall model evaluation summary)
# ---------------------------------------------------------------------------


def build_static_portrait_figure(
    metrics_frames: list[tuple[str, str, pd.DataFrame]],
    exclude_datasets: set[str] | None = None,
) -> Figure | None:
    """Overall-evaluation portrait plot, ESMValTool portrait_plot.py style.

    Models on x, variables on y, cell colour = RMSE normalized with
    ``centered_median`` across models ((rmse − median) / median), 11-step
    RdYlBu_r at ±0.5 ("Relative RMSE"), NaN cells white. With a single
    benchmarked model the centred normalization is meaningless, so the raw
    RMSE values are annotated on neutral cells instead.

    Args:
        metrics_frames: (tier_label, diagnostic_name, metrics_df) tuples
            collected from every rendered database.
        exclude_datasets: data_ids to drop from the columns (reference
            datasets — a portrait compares models, like portrait_plot.py).

    Returns
    -------
        The figure, or None if no usable weighted_rmse metrics were found.
    """
    import matplotlib as mpl

    rows = []
    for tier, _diag, mdf in metrics_frames:
        if (
            mdf is None
            or mdf.empty
            or "weighted_rmse" not in mdf.columns
            or "var_id" not in mdf.columns
            or "data_id" not in mdf.columns
        ):
            continue
        sub = mdf.dropna(subset=["weighted_rmse"])
        for rec in sub.itertuples(index=False):
            rows.append(
                {
                    "tier": tier,
                    "var_id": rec.var_id,
                    "data_id": rec.data_id,
                    "weighted_rmse": float(rec.weighted_rmse),
                },
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if exclude_datasets:
        df = df[~df["data_id"].isin(exclude_datasets)]
        if df.empty:
            return None

    # Disambiguate variables appearing in several tiers
    dup_vars = df.groupby("var_id")["tier"].nunique()
    dup_vars = set(dup_vars[dup_vars > 1].index)
    df["row_label"] = [
        f"{v} ({t})" if v in dup_vars else str(v)
        for v, t in zip(df["var_id"], df["tier"], strict=True)
    ]

    matrix = (
        df.pivot_table(
            index="row_label",
            columns="data_id",
            values="weighted_rmse",
            aggfunc="mean",
        )
        .dropna(how="all")
        .sort_index()
    )
    if matrix.empty:
        return None

    single_model = matrix.shape[1] < 2  # noqa: PLR2004
    if single_model:
        rel = matrix * 0.0
    else:
        median = matrix.median(axis=1)
        rel = matrix.sub(median, axis=0).div(median, axis=0)

    n_models = matrix.shape[1]
    n_vars = matrix.shape[0]
    with sns.axes_style(_SEABORN_STYLE):
        fig = Figure(
            figsize=(max(7.0, 0.4 * n_models + 3.0), max(3.5, 0.3 * n_vars + 1.5)),
            layout="constrained",
        )
        axes = fig.add_subplot(111)

    cmap = mpl.colormaps["RdYlBu_r"].resampled(11)
    cmap.set_bad("white")  # nan_color in portrait_plot.py
    data = np.ma.masked_invalid(rel.to_numpy(dtype=float))
    mappable = axes.imshow(
        data,
        cmap=cmap,
        vmin=-0.5,
        vmax=0.5,
        aspect="auto",
        interpolation="nearest",
    )
    axes.set_xticks(range(n_models))
    axes.set_xticklabels(matrix.columns, rotation=90, fontsize=_MAP_STATS_FONTSIZE)
    axes.set_yticks(range(n_vars))
    axes.set_yticklabels(matrix.index, fontsize=_MAP_STATS_FONTSIZE)
    # thin white grid between cells, like portrait_plot.py
    axes.set_xticks(np.arange(-0.5, n_models), minor=True)
    axes.set_yticks(np.arange(-0.5, n_vars), minor=True)
    axes.grid(which="minor", color="white", linewidth=1.0)
    axes.tick_params(which="minor", length=0)

    if single_model:
        for i in range(n_vars):
            val = matrix.iloc[i, 0]
            if np.isfinite(val):
                axes.annotate(
                    f"{val:.3g}",
                    xy=(0, i),
                    ha="center",
                    va="center",
                    fontsize=_MAP_STATS_FONTSIZE,
                )
        axes.set_title(
            "Overall model evaluation — RMSE vs reference (single model)",
            fontsize=_MAP_FONTSIZE + 2,
        )
    else:
        cbar = fig.colorbar(mappable, ax=axes, extend="both")
        cbar.set_label("Relative RMSE", fontsize=_MAP_FONTSIZE)
        cbar.ax.tick_params(labelsize=_MAP_FONTSIZE)
        axes.set_title("Overall model evaluation", fontsize=_MAP_FONTSIZE + 2)

    return fig


# ---------------------------------------------------------------------------
# Registry + export entry point
# ---------------------------------------------------------------------------

# Keyed by the detect_diag_type() strings, like _FIGURE_BUILDERS in
# _figures.py. Types without a static builder yet are skipped (with a log
# message) by generate_static_plots().
_STATIC_FIGURE_BUILDERS: dict[str, Callable] = {
    "AnnualCycle": build_static_annual_cycle_figures,
    "DiurnalCycle": build_static_diurnal_cycle_figures,
    "Histogram": build_static_histogram_figures,
    "Hovmoller": build_static_hovmoller_figures,
    "Map": build_static_map_figures,
    "QBO": build_static_qbo_figures,
    "Scalar": build_static_bar_figures,
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
    display_names: Mapping[str, str] | None = None,
) -> list[Path]:
    """Render static figures from result databases into *out_dir*.

    One file per (database, diagnostic, variable, format) is written to
    ``out_dir/<db_stem>/<diagnostic>/<var_id>.<format>``. When any diagnostic
    recorded weighted_rmse metrics, an overall portrait plot is additionally
    written to ``out_dir/portrait_plot.<format>``.

    Args:
        db_paths: Paths to one or more DuckDB .ddb files.
        out_dir: Output directory (created if missing).
        formats: Image formats understood by matplotlib (e.g. "png", "pdf").
        dpi: Resolution for raster formats (ESMValTool default: 300).
        display_names: Dataset id to the name shown in panel titles. Affects
            rendered text only; ids, metrics and file names are unchanged.

    Returns
    -------
        The list of written file paths.
    """
    written: list[Path] = []
    metrics_frames: list[tuple[str, str, pd.DataFrame]] = []
    reference_ids: set[str] = set()
    for db_path in db_paths:
        db_data = read_database(db_path)
        if not db_data:
            logger.warning("No diagnostic databases found in {}", db_path)
            continue
        for diag_name, tables in db_data.items():
            data_df = tables.get("raw_output", pd.DataFrame())
            metrics_df = tables.get("metrics", pd.DataFrame())
            variables_df = tables.get("variables", pd.DataFrame())
            if not metrics_df.empty:
                metrics_frames.append((db_path.stem, diag_name, metrics_df))
            if data_df.empty:
                continue
            if "data_type" in data_df.columns:
                reference_ids.update(
                    data_df.loc[
                        data_df["data_type"] == "reference",
                        "data_id",
                    ].unique(),
                )
            diag_type = detect_diag_type(data_df.columns.tolist())
            builder = _STATIC_FIGURE_BUILDERS.get(diag_type)
            if builder is None:
                logger.info(
                    "Skipping {} ({}): no static builder for this type yet",
                    diag_name,
                    diag_type,
                )
                continue
            # Only the map builder has titles worth overriding so far; give the
            # others the keyword when they grow one.
            if display_names and builder is build_static_map_figures:
                figures = builder(data_df, variables_df, metrics_df, display_names)
            else:
                figures = builder(data_df, variables_df, metrics_df)
            diag_dir = (
                out_dir / _safe_filename(db_path.stem) / _safe_filename(diag_name)
            )
            diag_dir.mkdir(parents=True, exist_ok=True)
            for var_id, fig in figures.items():
                base = diag_dir / _safe_filename(var_id)
                written.extend(_save_figure(fig, base, formats, dpi))

    portrait = build_static_portrait_figure(metrics_frames, reference_ids)
    if portrait is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        written.extend(_save_figure(portrait, out_dir / "portrait_plot", formats, dpi))

    logger.info("Wrote {} static figure file(s) to {}", len(written), out_dir)
    return written
