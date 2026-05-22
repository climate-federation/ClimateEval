"""HTTP server for the ClimateEval benchmarking report.

Reads DuckDB databases once on startup and serves Plotly figure JSON on
demand.  The browser receives a thin HTML shell (~few KB) and fetches each
plot only when the user selects a variable tab, making navigation fast
regardless of how many models or variables are in the database.

API endpoints
-------------
GET /                                   HTML shell (sidebar + empty panels).
GET /api/structure                      Tier / diagnostic / variable metadata (JSON).
GET /api/figure/<tier>/<diag>/<var>     Plotly figure for one variable (JSON).
GET /api/leaderboard                    Scoreboard + summary table (JSON).
GET /plotly.js                          Bundled Plotly.js for offline use.
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from climateeval.report._db import detect_diag_type, get_var_cols, read_database
from climateeval.report._figures import (
    _DIAG_ICONS,
    _DIAG_LABELS,
    _FIGURE_BUILDERS,
    _var_label,
)
from climateeval.report._html import build_html_shell, generate_portable_html
from climateeval.report._leaderboard import (
    build_leaderboard_data,
    build_scoreboard_figure,
)

# ---------------------------------------------------------------------------
# Global server state (populated once by _load_state before server starts)
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {
    "tiers": [],  # list[dict]  – see _load_state()
    "all_metrics": {},  # {tier_label: {diag_name: metrics_df}}
    "fig_cache": {},  # {(tier_id, diag_name, var_id): go.Figure}
    "html_shell": "",  # pre-rendered HTML shell served at GET /
    "lb_json": b"{}",  # pre-built leaderboard JSON bytes
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_state(
    db_paths: list[Path],
    model_name: str,
    user_name: str,
    developer: str,
) -> None:
    """Read all DuckDB files and build the server state."""
    tiers: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, pd.DataFrame]] = {}

    for tier_idx, db_path in enumerate(db_paths):
        from climateeval.report._html import _stem_to_label

        label = _stem_to_label(db_path.stem)
        db_data = read_database(db_path)
        if not db_data:
            print(f"[warn] No diagnostic data found in {db_path}", file=sys.stderr)
            continue

        tier_id = f"tier{tier_idx}"
        diag_types: dict[str, str] = {}
        tier_metrics: dict[str, pd.DataFrame] = {}

        for diag_name, tables in db_data.items():
            if "raw_output" not in tables:
                continue
            data_df: pd.DataFrame = tables["raw_output"]
            metrics_df: pd.DataFrame = tables.get("metrics", pd.DataFrame())
            dtype = detect_diag_type(data_df.columns.tolist())
            diag_types[diag_name] = dtype
            if not data_df.empty:
                tier_metrics[diag_name] = metrics_df

        tiers.append(
            {
                "id": tier_id,
                "label": label,
                "db_path": db_path,
                "db_data": db_data,
                "diag_types": diag_types,
            },
        )
        all_metrics[label] = tier_metrics

    _STATE["tiers"] = tiers
    _STATE["all_metrics"] = all_metrics
    _STATE["html_shell"] = build_html_shell(
        tiers,
        all_metrics,
        model_name,
        user_name,
        developer,
    )
    _STATE["lb_json"] = _build_leaderboard_json()


# ---------------------------------------------------------------------------
# Figure building (lazy, cached per (tier_id, diag_name))
# ---------------------------------------------------------------------------


def _get_figure_json(tier_id: str, diag_name: str, var_id: str) -> bytes | None:
    """Return Plotly figure JSON bytes for one variable (builds + caches all vars on first call)."""
    cache_key = (tier_id, diag_name, var_id)
    if cache_key in _STATE["fig_cache"]:
        return _STATE["fig_cache"][cache_key]

    # First request for this (tier, diag) pair: build all variables
    tier = next((t for t in _STATE["tiers"] if t["id"] == tier_id), None)
    if tier is None:
        return None
    tables = tier["db_data"].get(diag_name)
    if tables is None or "raw_output" not in tables:
        return None

    data_df: pd.DataFrame = tables["raw_output"]
    variables_df: pd.DataFrame = tables.get("variables", pd.DataFrame())
    metrics_df: pd.DataFrame = tables.get("metrics", pd.DataFrame())
    dtype = tier["diag_types"].get(diag_name, "Unknown")
    builder = _FIGURE_BUILDERS.get(dtype)
    if builder is None or data_df.empty:
        return None

    figures = builder(data_df, variables_df, metrics_df)
    for vid, fig in figures.items():
        _STATE["fig_cache"][(tier_id, diag_name, vid)] = fig.to_json().encode()

    return _STATE["fig_cache"].get(cache_key)


# ---------------------------------------------------------------------------
# Leaderboard JSON
# ---------------------------------------------------------------------------


def _build_leaderboard_json() -> bytes:
    """Return the leaderboard payload (scoreboard figure + summary rows) as JSON bytes."""
    payload: dict[str, Any] = {}

    scoreboard_fig = build_scoreboard_figure(_STATE["all_metrics"])
    if scoreboard_fig is not None:
        payload["scoreboard"] = json.loads(scoreboard_fig.to_json())

    summary_df, detail_df = build_leaderboard_data(_STATE["all_metrics"])
    if not summary_df.empty:
        payload["summary"] = json.loads(summary_df.to_json(orient="records"))
    if not detail_df.empty:
        metric_cols = [
            c
            for c in ["weighted_rmse", "weighted_pearsonr", "weighted_emd"]
            if c in detail_df.columns
        ]
        show_cols = [
            c
            for c in ["data_id", "tier", "diagnostic", "var_id", *metric_cols]
            if c in detail_df.columns
        ]
        payload["detail"] = json.loads(detail_df[show_cols].to_json(orient="records"))

    return json.dumps(payload).encode()


# ---------------------------------------------------------------------------
# Structure API
# ---------------------------------------------------------------------------


def _build_structure() -> bytes:
    """Return the tier / diagnostic / variable structure as JSON bytes."""
    tiers_out = []
    for tier in _STATE["tiers"]:
        diags_out = []
        for diag_name, tables in tier["db_data"].items():
            if "raw_output" not in tables:
                continue
            data_df: pd.DataFrame = tables["raw_output"]
            variables_df: pd.DataFrame = tables.get("variables", pd.DataFrame())
            dtype = tier["diag_types"].get(diag_name, "Unknown")

            variables = []
            for var_id in get_var_cols(data_df):
                if data_df[var_id].isna().all():
                    continue
                long_name, units = _var_label(var_id, variables_df)
                variables.append({"id": var_id, "long_name": long_name, "units": units})

            if not variables:
                continue

            diags_out.append(
                {
                    "name": diag_name,
                    "type": dtype,
                    "label": _DIAG_LABELS.get(diag_name)
                    or _DIAG_LABELS.get(dtype, diag_name),
                    "icon": _DIAG_ICONS.get(diag_name) or _DIAG_ICONS.get(dtype, ""),
                    "variables": variables,
                },
            )

        tiers_out.append(
            {"id": tier["id"], "label": tier["label"], "diagnostics": diags_out},
        )

    return json.dumps({"tiers": tiers_out}).encode()


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"  {self.address_string()} {fmt % args}", flush=True)

    def _send(
        self,
        body: bytes,
        content_type: str = "application/json",
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._send(_STATE["html_shell"].encode(), "text/html; charset=utf-8")
            return

        if path == "/api/leaderboard":
            self._send(_STATE["lb_json"])
            return

        if path == "/api/structure":
            self._send(_build_structure())
            return

        if path.startswith("/api/figure/"):
            # /api/figure/<tier_id>/<diag_name>/<var_id>
            parts = path[len("/api/figure/") :].split("/", 2)
            if len(parts) != 3:
                self._send(b'{"error":"bad request"}', status=400)
                return
            tier_id, diag_name, var_id = parts
            data = _get_figure_json(tier_id, diag_name, var_id)
            if data is None:
                self._send(b'{"error":"not found"}', status=404)
                return
            self._send(data)
            return

        if path == "/plotly.js":
            try:
                import plotly as _plotly

                js_path = (
                    Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
                )
                if js_path.exists():
                    self._send(js_path.read_bytes(), "application/javascript")
                    return
            except Exception:
                pass
            self._send(b"// plotly.js not available", "application/javascript", 404)
            return

        self._send(b'{"error":"not found"}', status=404)


# ---------------------------------------------------------------------------
# Public serve() function
# ---------------------------------------------------------------------------


def serve(
    db_paths: list[Path],
    *,
    model_name: str = "",
    user_name: str = "",
    developer: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    save_html: Path | None = None,
) -> None:
    """Load databases and start the local web server.

    Args:
        db_paths: Paths to one or more DuckDB .ddb files.
        model_name: Model label shown in the report header.
        user_name: Analyst / user name shown in the report header.
        developer: Benchmarking tool developer name.
        host: Server bind address.
        port: Server port.
        open_browser: If True, open the default web browser automatically.
        save_html: If a Path is given, also write a self-contained HTML file
                   to that location using generate_portable_html().
    """
    print("Loading databases…", flush=True)
    _load_state(db_paths, model_name, user_name, developer)
    n_tiers = len(_STATE["tiers"])
    print(
        f"  Loaded {n_tiers} tier(s): {[t['label'] for t in _STATE['tiers']]}",
        flush=True,
    )

    if save_html is not None:
        print(f"Writing portable HTML to {save_html}…", flush=True)
        html_content = generate_portable_html(
            db_paths,
            offline=True,
            model_name=model_name,
            user_name=user_name,
            developer=developer,
        )
        save_html.write_text(html_content, encoding="utf-8")
        print(f"  HTML saved to {save_html}", flush=True)

    url = f"http://{host}:{port}"
    print(f"\nServing at {url}", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    server = HTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
