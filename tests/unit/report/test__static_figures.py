from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import pandas as pd

from climateeval.report._figures import _FIGURE_BUILDERS
from climateeval.report._static_figures import (
    _STATIC_FIGURE_BUILDERS,
    build_static_map_figures,
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


def test_registry_keys_are_valid_diag_types() -> None:
    assert set(_STATIC_FIGURE_BUILDERS) <= set(_FIGURE_BUILDERS)
