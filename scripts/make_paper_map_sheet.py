"""Compose paper map sheets (model | observations | bias) from a scored .ddb.

Usage: make_paper_map_sheet.py <database.ddb> <output-dir>
       make_paper_map_sheet.py --table <fields.npz> <output.png>

The second form draws the SAME sheet from a gridded table instead of a scored
database, for windows ClimateEval has not scored yet (a partial month against
the reference's calendar-month climatology).  The npz holds ``lat``, ``lon``
(1-D, degrees), ``model`` (str), ``title`` (str), ``fields`` (list of var ids)
and per field ``model.<var>``, ``ref.<var>`` (2-D, lat x lon), ``ref_name.<var>``,
``units.<var>``, ``long_name.<var>``.  No metrics: the bias panel's mean and RMSE
are computed from the grids by the sheet builder itself.

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
    z = np.load(npz_path, allow_pickle=True)
    lat, lon = np.asarray(z["lat"], float), np.asarray(z["lon"], float)
    fields = [str(v) for v in z["fields"]]
    model = str(z["model"])
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    base = {"latitude": la.ravel(), "longitude": lo.ravel()}
    rows = [pd.DataFrame({**base, "data_id": model, "data_type": "to_benchmark",
                          **{v: np.asarray(z[f"model.{v}"], float).ravel() for v in fields}})]
    for ref in sorted({str(z[f"ref_name.{v}"]) for v in fields}):
        cols = {v: np.asarray(z[f"ref.{v}"], float).ravel() for v in fields
                if str(z[f"ref_name.{v}"]) == ref}
        rows.append(pd.DataFrame({**base, "data_id": ref, "data_type": "reference", **cols}))
    data_df = pd.concat(rows, ignore_index=True)
    variables_df = pd.DataFrame({"id": fields,
                                 "long_name": [str(z[f"long_name.{v}"]) for v in fields],
                                 "units": [str(z[f"units.{v}"]) for v in fields]})
    return data_df, variables_df, fields, model, str(z["title"])


if len(sys.argv) > 1 and sys.argv[1] == "--table":
    data_df, variables_df, fields, model, title = frames_from_table(Path(sys.argv[2]))
    fig = build_static_map_sheet(data_df, variables_df, fields, display_names={model: model})
    if fig is None:
        sys.exit("no sheet: no field had both a model and a reference")
    fig.suptitle(title, fontsize=9)
    out = Path(sys.argv[3]); fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
    sys.exit(0)

from climateeval.report._db import read_database  # noqa: E402

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
