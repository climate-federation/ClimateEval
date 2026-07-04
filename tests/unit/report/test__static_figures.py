from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import pandas as pd

from climateeval.report._static_figures import (
    _STATIC_FIGURE_BUILDERS,
    build_static_timeseries_figures,
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
    from climateeval.report._figures import _FIGURE_BUILDERS

    assert set(_STATIC_FIGURE_BUILDERS) <= set(_FIGURE_BUILDERS)
