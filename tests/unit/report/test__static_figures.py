from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import pandas as pd
import pytest

from climateeval.report._figures import _FIGURE_BUILDERS
from climateeval.report._static_figures import (
    _STATIC_FIGURE_BUILDERS,
    build_static_annual_cycle_figures,
    build_static_bar_figures,
    build_static_depth_profile_figures,
    build_static_diurnal_cycle_figures,
    build_static_histogram_figures,
    build_static_hovmoller_figures,
    build_static_map_figures,
    build_static_portrait_figure,
    build_static_qbo_figures,
    build_static_timeseries_figures,
    build_static_zonal_line_figures,
    build_static_zonal_profile_figures,
    generate_static_plots,
)

if TYPE_CHECKING:
    from pathlib import Path

VARIABLES_DF = pd.DataFrame(
    [
        {"id": "tas", "long_name": "Near-Surface Air Temperature", "units": "K"},
        {"id": "pr", "long_name": "Precipitation", "units": "mm day-1"},
    ],
)


def _timeseries_df(datasets: dict[str, str], n_times: int = 5) -> pd.DataFrame:
    """Yearly rows for each dataset; datasets maps data_id → data_type."""
    rows = [
        {
            "time": pd.Timestamp(f"{2000 + t}-07-01"),
            "tas": 287.0 + offset + 0.1 * t,
            "data_id": ds,
            "data_type": dtype,
        }
        for offset, (ds, dtype) in enumerate(datasets.items())
        for t in range(n_times)
    ]
    return pd.DataFrame(rows)


def test_timeseries_figures_one_per_variable_with_iconeval_styles() -> None:
    data_df = _timeseries_df(
        {"ERA5": "reference", "MODEL": "to_benchmark", "CMIP6-X": "other"},
    )
    figures = build_static_timeseries_figures(data_df, VARIABLES_DF)

    assert set(figures) == {"tas"}
    fig = figures["tas"]
    assert len(fig.axes) == 1
    axes = fig.axes[0]
    lines = {line.get_label(): line for line in axes.get_lines()}

    ref = lines["ERA5"]
    assert ref.get_color() == "black"
    assert ref.get_linewidth() == 1.0

    model = lines["MODEL"]
    assert model.get_color() != "black"
    assert model.get_linewidth() == 1.25
    assert model.get_zorder() > ref.get_zorder()

    # Auxiliary datasets: thin light-gray background bundle, no legend entry.
    other = next(ln for ln in axes.get_lines() if ln.get_color() == "lightgray")
    assert other.get_linewidth() == 0.75
    legend_labels = [t.get_text() for t in axes.get_legend().get_texts()]
    assert legend_labels == ["MODEL", "ERA5"]

    assert axes.get_title() == "Near-Surface Air Temperature"
    assert axes.get_ylabel() == "tas [K]"


def test_timeseries_second_reference_gets_distinct_linestyle() -> None:
    data_df = _timeseries_df(
        {"ERA5": "reference", "HadCRUT5": "reference", "MODEL": "to_benchmark"},
    )
    figures = build_static_timeseries_figures(data_df, VARIABLES_DF)
    lines = {ln.get_label(): ln for ln in figures["tas"].axes[0].get_lines()}
    assert lines["ERA5"].get_linestyle() != lines["HadCRUT5"].get_linestyle()


def test_timeseries_few_points_use_markers() -> None:
    data_df = _timeseries_df({"MODEL": "to_benchmark"}, n_times=2)
    figures = build_static_timeseries_figures(data_df, VARIABLES_DF)
    (line,) = figures["tas"].axes[0].get_lines()
    assert line.get_marker() == "o"


def _map_df(datasets: dict[str, str]) -> pd.DataFrame:
    """Lat/lon grid rows for each dataset; datasets maps data_id → data_type."""
    rows = [
        {
            "latitude": lat,
            "longitude": lon,
            "tas": 280.0 + offset + lat / 10.0,
            "data_id": ds,
            "data_type": dtype,
        }
        for offset, (ds, dtype) in enumerate(datasets.items())
        for lat in (-30.0, 0.0, 30.0)
        for lon in (0.0, 90.0, 180.0, 270.0)
    ]
    return pd.DataFrame(rows)


def test_map_single_model_uses_esmvaltool_three_panel_layout() -> None:
    data_df = _map_df({"ERA5": "reference", "MODEL": "to_benchmark"})
    metrics_df = pd.DataFrame(
        [
            {
                "var_id": "tas",
                "data_id": "MODEL",
                "weighted_rmse": 1.23,
                "weighted_pearsonr": 0.98,
            },
        ],
    )
    figures = build_static_map_figures(data_df, VARIABLES_DF, metrics_df)

    fig = figures["tas"]
    # 3 map panels + 1 common colorbar + 1 bias colorbar
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines")]
    assert len(map_axes) == 3
    assert len(fig.axes) - len(map_axes) == 2
    titles = {ax.get_title() for ax in map_axes}
    assert titles == {"MODEL", "ERA5", "MODEL \u2212 ERA5"}
    bias_ax = next(ax for ax in map_axes if "\u2212" in ax.get_title())
    stats = [t.get_text() for t in bias_ax.texts]
    assert any("RMSE=1.23" in s for s in stats)
    # R² is computed from the gridded fields (multi_datasets.py definition)
    assert any(s.startswith("R²=") for s in stats)


def test_map_multi_model_grid_has_reference_and_bias_panels() -> None:
    data_df = _map_df(
        {"ERA5": "reference", "M1": "to_benchmark", "M2": "to_benchmark"},
    )
    figures = build_static_map_figures(data_df, VARIABLES_DF)
    fig = figures["tas"]
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines")]
    titles = [ax.get_title() for ax in map_axes]
    assert titles == ["ERA5", "M1", "M2"]
    assert "bias vs ERA5" in fig.get_suptitle()


def test_map_without_reference_renders_absolute_grid() -> None:
    data_df = _map_df({"M1": "to_benchmark", "M2": "to_benchmark"})
    figures = build_static_map_figures(data_df, VARIABLES_DF)
    fig = figures["tas"]
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines")]
    assert [ax.get_title() for ax in map_axes] == ["M1", "M2"]
    # single shared colorbar for the absolute panels
    assert len(fig.axes) - len(map_axes) == 1


def test_zonal_line_draws_one_line_per_dataset_vs_latitude() -> None:
    rows = [
        {
            "latitude": lat,
            "tas": 280.0 + off - abs(lat) / 3.0,
            "data_id": ds,
            "data_type": dt,
        }
        for off, (ds, dt) in enumerate(
            {"ERA5": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for lat in (-60.0, -30.0, 0.0, 30.0, 60.0)
    ]
    figures = build_static_zonal_line_figures(pd.DataFrame(rows), VARIABLES_DF)
    axes = figures["tas"].axes[0]
    assert len(axes.get_lines()) == 2
    assert axes.get_xlabel() == "latitude [°N]"


def test_depth_profile_draws_one_line_per_dataset_surface_on_top() -> None:
    rows = [
        {
            "depth": depth,
            "tas": 20.0 + off - depth / 200.0,
            "data_id": ds,
            "data_type": dt,
        }
        for off, (ds, dt) in enumerate(
            {"IAP": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for depth in (0.0, 100.0, 500.0, 1000.0, 2000.0)
    ]
    figures = build_static_depth_profile_figures(pd.DataFrame(rows), VARIABLES_DF)
    axes = figures["tas"].axes[0]
    assert len(axes.get_lines()) == 2
    assert axes.get_ylabel() == "depth [m]"
    assert axes.get_xlabel() == "tas [K]"
    assert axes.yaxis_inverted()


def _profile_df(datasets: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "latitude": lat,
                "air_pressure": plev,
                "ta": 220.0 + off + plev / 50.0,
                "data_id": ds,
                "data_type": dt,
            }
            for off, (ds, dt) in enumerate(datasets.items())
            for lat in (-30.0, 0.0, 30.0)
            for plev in (1000.0, 500.0, 100.0)
        ],
    )


def test_zonal_profile_single_model_three_panels_log_pressure() -> None:
    variables = pd.DataFrame(
        [{"id": "ta", "long_name": "Air Temperature", "units": "K"}],
    )
    figures = build_static_zonal_profile_figures(
        _profile_df({"ERA5": "reference", "MODEL": "to_benchmark"}),
        variables,
    )
    fig = figures["ta"]
    panel_axes = [ax for ax in fig.axes if ax.get_yscale() == "log"]
    # 3 profile panels; colorbar axes are not log-scaled
    assert len(panel_axes) == 3
    assert len(fig.axes) - len(panel_axes) == 2
    for ax in panel_axes:
        ymin, ymax = ax.get_ylim()
        assert ymin > ymax  # pressure decreases upward


def test_annual_cycle_months_on_x_axis() -> None:
    rows = [
        {
            "month_number": m,
            "tas": 285.0 + off + m / 10.0,
            "data_id": ds,
            "data_type": dt,
        }
        for off, (ds, dt) in enumerate(
            {"ERA5": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for m in range(1, 13)
    ]
    figures = build_static_annual_cycle_figures(pd.DataFrame(rows), VARIABLES_DF)
    axes = figures["tas"].axes[0]
    assert len(axes.get_lines()) == 2
    assert axes.get_xlabel() == "month"
    assert [t.get_text() for t in axes.get_xticklabels()][:3] == ["J", "F", "M"]


def test_diurnal_cycle_hours_on_x_axis() -> None:
    rows = [
        {
            "hour": h,
            "tas": 285.0 + h / 10.0,
            "data_id": "MODEL",
            "data_type": "to_benchmark",
        }
        for h in range(24)
    ]
    figures = build_static_diurnal_cycle_figures(pd.DataFrame(rows), VARIABLES_DF)
    assert figures["tas"].axes[0].get_xlabel() == "hour"


def test_histogram_bars_with_log_density_for_precip() -> None:
    rows = [
        {
            "pr": center,
            "density_pr": 10.0 ** (-center) + off,
            "data_id": ds,
            "data_type": dt,
        }
        for off, (ds, dt) in enumerate(
            {"GPCP": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for center in (0.5, 1.5, 2.5, 3.5)
    ]
    figures = build_static_histogram_figures(pd.DataFrame(rows), VARIABLES_DF)
    axes = figures["pr"].axes[0]
    assert axes.get_yscale() == "log"
    assert axes.get_ylabel() == "Density"
    assert len(axes.patches) == 8  # 4 bars x 2 datasets
    legend_labels = [t.get_text() for t in axes.get_legend().get_texts()]
    assert legend_labels == ["MODEL", "GPCP"]


def test_hovmoller_stacked_panels_share_one_colorbar() -> None:
    rows = [
        {
            "time": pd.Timestamp(f"2000-{m:02d}-15"),
            "longitude": lon,
            "pr": (m - 6.5) * (1 + off),
            "data_id": ds,
            "data_type": dt,
        }
        for off, (ds, dt) in enumerate(
            {"GPCP": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for m in range(1, 13)
        for lon in (0.0, 90.0, 180.0, 270.0)
    ]
    figures = build_static_hovmoller_figures(pd.DataFrame(rows), VARIABLES_DF)
    fig = figures["pr"]
    panel_axes = [ax for ax in fig.axes if ax.get_title()]
    assert [ax.get_title() for ax in panel_axes] == ["MODEL", "GPCP"]
    assert len(fig.axes) - len(panel_axes) == 1  # one shared colorbar
    for ax in panel_axes:
        ymin, ymax = ax.get_ylim()
        assert ymin > ymax  # time increases downward


def test_qbo_layout_heatmaps_amplitude_and_index() -> None:
    variables = pd.DataFrame(
        [{"id": "ua", "long_name": "Eastward Wind", "units": "m s-1"}],
    )
    rows = [
        {
            "time": pd.Timestamp(f"200{y}-{m:02d}-15"),
            "air_pressure": plev,
            "ua": 10.0 * (1 if (m + y) % 2 else -1),
            "data_id": ds,
            "data_type": dt,
        }
        for ds, dt in {"ERA5": "reference", "MODEL": "to_benchmark"}.items()
        for y in range(2)
        for m in range(1, 13)
        for plev in (10000.0, 5000.0, 1000.0)
    ]
    figures = build_static_qbo_figures(pd.DataFrame(rows), variables)
    fig = figures["ua"]
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert "MODEL" in titles
    assert "ERA5" in titles
    assert "Amplitude" in titles
    assert any("QBO index" in t for t in titles)


def _write_sample_db(db_path: Path) -> None:
    """Create a minimal suite-shaped database with one TimeSeries diagnostic."""
    conn = ibis.duckdb.connect(db_path)
    conn.create_database("timeseries_tas")
    data_df = _timeseries_df({"ERA5": "reference", "MODEL": "to_benchmark"})
    conn.create_table("raw_output", data_df, database="timeseries_tas")
    conn.create_table("variables", VARIABLES_DF, database="timeseries_tas")
    conn.disconnect()


def test_generate_static_plots_writes_requested_formats(tmp_path: Path) -> None:
    db_path = tmp_path / "Tier2_test.ddb"
    _write_sample_db(db_path)
    out_dir = tmp_path / "figs"

    written = generate_static_plots(
        [db_path],
        out_dir,
        formats=("png", "pdf"),
        dpi=72,
    )

    expected = {
        out_dir / "Tier2_test" / "timeseries_tas" / "tas.png",
        out_dir / "Tier2_test" / "timeseries_tas" / "tas.pdf",
    }
    assert set(written) == expected
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0


def test_scalar_bars_colored_by_data_type_with_benchmark_on_top() -> None:
    rows = [
        {"data_id": "MODEL", "data_type": "to_benchmark", "tas": 288.1},
        {"data_id": "ERA5", "data_type": "reference", "tas": 287.9},
        {"data_id": "CMIP6-X", "data_type": "other", "tas": 287.5},
    ]
    figures = build_static_bar_figures(pd.DataFrame(rows), VARIABLES_DF)
    axes = figures["tas"].axes[0]
    labels = [t.get_text() for t in axes.get_yticklabels()]
    assert labels[0] == "MODEL"  # first y tick = top bar
    assert len(axes.patches) == 3
    annotations = [t.get_text() for t in axes.texts]
    assert "288" in "".join(annotations)


def _metrics_df(models: dict[str, float], var_id: str = "tas") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"var_id": var_id, "data_id": m, "weighted_rmse": v}
            for m, v in models.items()
        ],
    )


def test_portrait_multi_model_centered_median_normalization() -> None:
    frames = [
        ("Tier2", "map", _metrics_df({"M1": 1.0, "M2": 2.0, "M3": 3.0})),
        ("Tier2", "map", _metrics_df({"M1": 4.0, "M2": 5.0, "M3": 6.0}, "pr")),
    ]
    fig = build_static_portrait_figure(frames)
    assert fig is not None
    axes = fig.axes[0]
    assert [t.get_text() for t in axes.get_xticklabels()] == ["M1", "M2", "M3"]
    assert [t.get_text() for t in axes.get_yticklabels()] == ["pr", "tas"]
    img = axes.get_images()[0]
    data = img.get_array()
    # centered_median: median model → 0
    assert data[1, 1] == 0.0  # tas row, M2 = median
    assert data[1, 0] == -0.5  # (1 - 2) / 2
    assert img.get_clim() == (-0.5, 0.5)


def test_portrait_single_model_annotates_raw_rmse() -> None:
    fig = build_static_portrait_figure(
        [("Tier2", "map", _metrics_df({"MODEL": 1.23}))],
    )
    assert fig is not None
    axes = fig.axes[0]
    assert any("1.23" in t.get_text() for t in axes.texts)
    assert "single model" in axes.get_title()


def test_portrait_excludes_reference_datasets_and_empty_rows() -> None:
    frames = [
        ("Tier2", "map", _metrics_df({"M1": 1.0, "M2": 2.0, "ERA5": 0.5})),
        # row with metrics only for the excluded reference → dropped entirely
        ("Tier2", "map", _metrics_df({"ERA5": 0.4}, "pr")),
    ]
    fig = build_static_portrait_figure(frames, exclude_datasets={"ERA5"})
    assert fig is not None
    axes = fig.axes[0]
    assert [t.get_text() for t in axes.get_xticklabels()] == ["M1", "M2"]
    assert [t.get_text() for t in axes.get_yticklabels()] == ["tas"]


def test_portrait_returns_none_without_metrics() -> None:
    assert build_static_portrait_figure([]) is None
    assert (
        build_static_portrait_figure([("T", "d", pd.DataFrame({"other": [1]}))]) is None
    )


def test_registry_covers_all_interactive_figure_types() -> None:
    assert set(_STATIC_FIGURE_BUILDERS) == set(_FIGURE_BUILDERS)


def _sheet_df() -> pd.DataFrame:
    """Two variables, one model and one reference, on a small lat/lon grid."""
    rows = [
        {
            "data_id": ds,
            "data_type": dtype,
            "latitude": lat,
            "longitude": lon,
            "tas": 280.0 + offset,
            "pr": 2.0 + offset,
        }
        for offset, (ds, dtype) in enumerate(
            {"ERA5": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for lat in (-30.0, 0.0, 30.0)
        for lon in (0.0, 90.0, 180.0, 270.0)
    ]
    return pd.DataFrame(rows)


def test_map_sheet_uses_display_names_without_touching_ids() -> None:
    from climateeval.report._static_figures import build_static_map_sheet

    data_df = _sheet_df()
    fig = build_static_map_sheet(
        data_df,
        VARIABLES_DF,
        ["tas", "pr"],
        display_names={"MODEL": "ESM", "ERA5": "ERA5-obs"},
    )
    assert fig is not None
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    joined = " ".join(titles)
    assert "ESM" in joined
    assert "ERA5-obs" in joined
    # The override is presentation only: the frame keeps the real identifiers.
    assert set(data_df["data_id"]) == {"ERA5", "MODEL"}
    assert "MODEL" not in joined


def test_map_sheet_refuses_rows_with_different_models() -> None:
    from climateeval.report._static_figures import build_static_map_sheet

    data_df = _sheet_df()
    # Give "pr" a second model so the two rows disagree on whose data is drawn.
    other = data_df[data_df["data_id"] == "MODEL"].copy()
    other["data_id"] = "MODEL-B"
    other["tas"] = float("nan")
    data_df = pd.concat([data_df, other], ignore_index=True)
    data_df.loc[data_df["data_id"] == "MODEL", "pr"] = float("nan")

    with pytest.raises(ValueError, match="one model across all rows"):
        build_static_map_sheet(data_df, VARIABLES_DF, ["tas", "pr"])
