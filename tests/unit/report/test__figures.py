from __future__ import annotations

import re

import pandas as pd

from climateeval.report._figures import (
    build_map_figures,
    build_qbo_figures,
    build_zonal_profile_figures,
)

VARIABLES_DF = pd.DataFrame(
    [
        {"id": "tas", "long_name": "Near-Surface Air Temperature", "units": "K"},
        {"id": "ta", "long_name": "Air Temperature", "units": "K"},
        {"id": "ua", "long_name": "Eastward Wind", "units": "m s-1"},
    ],
)


def _axis_layout(fig: object, prefix: str) -> dict[str, dict]:
    """Return {axis_name: axis_dict} for all x or y axes of a figure."""
    layout = fig.layout.to_plotly_json()  # type: ignore[attr-defined]
    return {
        key: val for key, val in layout.items() if re.fullmatch(rf"{prefix}\d*", key)
    }


def _assert_all_axes_matched(fig: object) -> None:
    x_axes = _axis_layout(fig, "xaxis")
    y_axes = _axis_layout(fig, "yaxis")
    assert x_axes
    assert y_axes
    for name, ax in x_axes.items():
        assert ax.get("matches") == "x", f"{name} not matched"
    for name, ax in y_axes.items():
        assert ax.get("matches") == "y", f"{name} not matched"


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


def test_map_figures_bias_layout_syncs_all_panels() -> None:
    data_df = _map_df({"ERA5": "reference", "MODEL": "to_benchmark"})
    figures = build_map_figures(data_df, VARIABLES_DF)
    fig = figures["tas"]
    # reference | model | bias columns
    assert len(_axis_layout(fig, "xaxis")) == 3
    _assert_all_axes_matched(fig)


def test_map_figures_no_reference_grid_syncs_all_panels() -> None:
    data_df = _map_df({"MODEL-A": "to_benchmark", "MODEL-B": "to_benchmark"})
    figures = build_map_figures(data_df, VARIABLES_DF)
    fig = figures["tas"]
    assert len(_axis_layout(fig, "xaxis")) == 2
    _assert_all_axes_matched(fig)


def test_zonal_profile_figures_sync_all_panels() -> None:
    rows = [
        {
            "latitude": lat,
            "air_pressure": plev,
            "ta": 250.0 + offset + plev / 10000.0,
            "data_id": ds,
            "data_type": dtype,
        }
        for offset, (ds, dtype) in enumerate(
            {"ERA5": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for lat in (-30.0, 0.0, 30.0)
        for plev in (100000.0, 50000.0, 10000.0)
    ]
    figures = build_zonal_profile_figures(pd.DataFrame(rows), VARIABLES_DF)
    fig = figures["ta"]
    # reference | bias columns
    assert len(_axis_layout(fig, "xaxis")) == 2
    _assert_all_axes_matched(fig)


def test_qbo_figures_sync_heatmap_rows_only() -> None:
    times = pd.date_range("2000-01-01", periods=24, freq="MS")
    rows = [
        {
            "time": t,
            "air_pressure": plev,
            "ua": (-1.0) ** i + offset,
            "data_id": ds,
            "data_type": dtype,
        }
        for offset, (ds, dtype) in enumerate(
            {"ERA5": "reference", "MODEL": "to_benchmark"}.items(),
        )
        for i, t in enumerate(times)
        for plev in (10000.0, 5000.0, 1000.0)
    ]
    figures = build_qbo_figures(pd.DataFrame(rows), VARIABLES_DF)
    fig = figures["ua"]
    x_axes = _axis_layout(fig, "xaxis")
    y_axes = _axis_layout(fig, "yaxis")
    # Subplot order: (1,1) heatmap 1 → x/y, (1,2) amplitude → x2/y2,
    # (2,1) heatmap 2 → x3/y3, (3,1) QBO index → x4/y4.
    assert set(x_axes) == {"xaxis", "xaxis2", "xaxis3", "xaxis4"}
    assert x_axes["xaxis3"].get("matches") == "x"
    assert y_axes["yaxis3"].get("matches") == "y"
    # Amplitude and index panels keep independent axes.
    for name in ("xaxis2", "xaxis4"):
        assert "matches" not in x_axes[name]
    for name in ("yaxis2", "yaxis4"):
        assert "matches" not in y_axes[name]
