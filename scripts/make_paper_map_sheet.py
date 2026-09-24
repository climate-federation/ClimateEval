"""Compose paper map sheets (model | observations | bias) from a scored .ddb.

Usage: make_paper_map_sheet.py <database.ddb> <output-dir>
       make_paper_map_sheet.py --table <fields.npz> <output.png> [dpi]

The second form draws the SAME sheet from a gridded table instead of a scored
database, for windows ClimateEval has not scored yet (a partial month against
the reference's calendar-month climatology).  The npz holds ``lat``, ``lon``
(1-D, degrees), ``model`` (str), ``title`` (str), ``window`` (str: the model
averaging window, rendered into the title mechanically so the partial-window
caveat cannot be left out), ``fields`` (list of var ids)
and per field ``model.<var>``, ``ref.<var>`` (2-D, lat x lon), ``ref_name.<var>``,
``units.<var>``, ``long_name.<var>``.  No metrics: the bias panel's mean and RMSE
are computed from the grids by the sheet builder itself.  The table form checks
what the database form gets from ESMValTool: every grid is (lat, lon) on a
regular, duplicate-free lat/lon grid, the model id differs from every reference
name, an all-missing grid is refused, and model and reference are both masked
to their COMMON valid cells (refused when empty) so all three panels and their
means use the same cells.  Units are
labels only: the producer must supply model and reference in the same unit.
The default 600 dpi is for the paper; a checkpoint figure passes a lower dpi.

Writes a 600 dpi PNG and a vector PDF per field set.  The model is displayed
as "ESM"; that is a presentation override only, so the identifiers in the
database, the metrics and the scoring are untouched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from climateeval.report._static_figures import build_static_map_sheet


def frames_from_table(npz_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str]:
    """Long data table + variables table from the gridded npz described above."""
    z = np.load(npz_path)
    if np.asarray(z["lat"]).ndim != 1 or np.asarray(z["lon"]).ndim != 1:
        raise ValueError("lat and lon must be 1-D axes")
    lat, lon = np.asarray(z["lat"], float), np.asarray(z["lon"], float) % 360.0
    if lat.min() < -90.0 or lat.max() > 90.0:
        raise ValueError("lat must lie in [-90, 90] degrees")
    for name, ax in (("lat", lat), ("lon", lon)):
        if ax.size < 2 or np.unique(ax).size != ax.size:
            raise ValueError(f"{name} must be 1-D without duplicates (a 0/360 pair counts)")
        step = np.diff(np.sort(ax))
        if not np.allclose(step, step[0], rtol=1e-3):
            raise ValueError(f"{name} must be regularly spaced: cos-lat weighting assumes it")
    fields = [str(v) for v in z["fields"]]
    model = str(z["model"])
    refs = {v: str(z[f"ref_name.{v}"]) for v in fields}
    if model in refs.values():
        raise ValueError(f"model id {model!r} collides with a reference name")
    grids = {}
    for v in fields:
        m, r = np.asarray(z[f"model.{v}"], float), np.asarray(z[f"ref.{v}"], float)
        for label, g in (("model", m), ("ref", r)):
            if g.shape != (lat.size, lon.size):
                raise ValueError(f"{label}.{v} has shape {g.shape}, expected (lat, lon) = "
                                 f"{(lat.size, lon.size)}")
            if not np.isfinite(g).any():
                raise ValueError(f"{label}.{v} is entirely missing")
        common = np.isfinite(m) & np.isfinite(r)
        if not common.any():
            raise ValueError(f"{v}: model and reference have no valid cell in common")
        grids[v] = (np.where(common, m, np.nan), np.where(common, r, np.nan))  # same cells everywhere
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    base = {"latitude": la.ravel(), "longitude": lo.ravel()}
    rows = [pd.DataFrame({**base, "data_id": model, "data_type": "to_benchmark",
                          **{v: grids[v][0].ravel() for v in fields}})]
    for ref in sorted(set(refs.values())):
        cols = {v: grids[v][1].ravel() for v in fields if refs[v] == ref}
        rows.append(pd.DataFrame({**base, "data_id": ref, "data_type": "reference", **cols}))
    data_df = pd.concat(rows, ignore_index=True)
    variables_df = pd.DataFrame({"id": fields,
                                 "long_name": [str(z[f"long_name.{v}"]) for v in fields],
                                 "units": [str(z[f"units.{v}"]) for v in fields]})
    title = (f"{z['title']}\n{model} averaging window: {z['window']}; middle panels: "
             f"reference calendar-month climatology ({', '.join(sorted(set(refs.values())))})")
    return data_df, variables_df, fields, model, title


if len(sys.argv) > 1 and sys.argv[1] == "--table":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    data_df, variables_df, fields, model, title = frames_from_table(Path(sys.argv[2]))
    fig = build_static_map_sheet(data_df, variables_df, fields, display_names={model: model})
    if fig is None:
        sys.exit("no sheet: no field had both a model and a reference")
    fig.suptitle(title, fontsize=9)
    out = Path(sys.argv[3]); dpi = int(sys.argv[4]) if len(sys.argv) == 5 else 600
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); print("wrote", out)
    sys.exit(0)

from climateeval.report._db import read_database  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
DB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FIELDS_6 = ["clt", "rsut", "swcre", "tas", "pr", "prw"]
# The coordinator's "sound" set, kept so the requested artifact exists.
FIELDS_4 = ["clt", "rsut", "swcre", "pr"]
# Measured from the CMOR cell_methods rather than assumed: clt, tas and prw
# are all published as "time: point within days", i.e. a fixed-hour snapshot
# averaged over the month, so their bias panels alias the diurnal cycle.
# rsut, rsutcs, rlut and pr carry "area: time: mean", so swcre (built from
# rsut and rsutcs) is sound too.  Cloud cover IS affected, which the
# four-field set above does not reflect.
FIELDS_SOUND = ["rsut", "swcre", "pr"]

db = read_database(DB)
tables = db["Map"] if "Map" in db else next(
    t for name, t in db.items() if "map" in name.lower()
)
data_df, metrics_df, variables_df = (
    tables["raw_output"], tables.get("metrics"), tables["variables"]
)
model_ids = sorted(set(data_df["data_id"]) - set(
    data_df.loc[data_df["data_type"] == "reference", "data_id"]
))
names = {m: "ESM" for m in model_ids}
# Strip the source-class prefix for print: "observation_CERES-EBAF" reads as
# furniture in a paper, "CERES-EBAF" is the dataset a reader looks up.
for ds in data_df["data_id"].unique():
    ds = str(ds)
    for prefix in ("observation_", "reanalysis_"):
        if ds.startswith(prefix):
            names[ds] = ds[len(prefix):]
print("model ids ->", names)

for fields, suffix in (
    (FIELDS_6, "six"), (FIELDS_4, "four"), (FIELDS_SOUND, "sound")
):
    fig = build_static_map_sheet(
        data_df, variables_df, fields,
        metrics_df=metrics_df, display_names=names,
    )
    if fig is None:
        print("no sheet for", suffix)
        continue
    path = OUT / f"cam6_feb_sheet_{suffix}_fields.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print("wrote", path)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", path.with_suffix(".pdf"))
