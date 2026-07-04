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

from climateeval.report._common import (
    _get_ref_datasets,
    _sort_datasets,
    _var_label,
)
from climateeval.report._db import detect_diag_type, get_var_cols, read_database
from climateeval.report._style import _SEABORN_COLORS

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
# Registry + export entry point
# ---------------------------------------------------------------------------

# Keyed by the detect_diag_type() strings, like _FIGURE_BUILDERS in
# _figures.py. Types without a static builder yet are skipped (with a log
# message) by generate_static_plots().
_STATIC_FIGURE_BUILDERS: dict[str, Callable] = {
    "TimeSeries": build_static_timeseries_figures,
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
            diag_dir = out_dir / _safe_filename(db_path.stem) / _safe_filename(
                diag_name
            )
            diag_dir.mkdir(parents=True, exist_ok=True)
            for var_id, fig in figures.items():
                base = diag_dir / _safe_filename(var_id)
                written.extend(_save_figure(fig, base, formats, dpi))
    logger.info("Wrote {} static figure file(s) to {}", len(written), out_dir)
    return written
