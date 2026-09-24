"""Compose paper map sheets (model | observations | bias) from a scored .ddb.

Usage: make_paper_map_sheet.py <database.ddb> <output-dir>

Writes a 600 dpi PNG and a vector PDF per field set.  The model is displayed
as "ESM"; that is a presentation override only, so the identifiers in the
database, the metrics and the scoring are untouched.
"""
import sys
from pathlib import Path

from climateeval.report._db import read_database
from climateeval.report._static_figures import build_static_map_sheet

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
