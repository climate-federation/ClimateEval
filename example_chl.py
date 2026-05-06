"""
Generate ClimateEval database for CMIP6 Chl-a vs ESA-CCI.
Models: GFDL-ESM4, CESM2, MPI-ESM1-2-LR
Analysis: global + 5 ocean basins
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
import iris.analysis
import numpy as np
from iris import Constraint
from iris.util import equalise_attributes, unify_time_units
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

# ============================================================
# Logger
# ============================================================
logger = logger.opt(colors=True)
logger.remove()
logger.add(sys.stdout, level="DEBUG",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
           colorize=True)

# ============================================================
# Paths
# ============================================================
suite_name    = "Tier2_ocean"
data_root_dir = Path("/work/bd1179/b384036/climateeval_input")
data_root_dir.mkdir(parents=True, exist_ok=True)

db_path = Path.home() / f"{suite_name}.ddb"
if db_path.is_file():
    db_path.unlink()
    logger.info(f"Removed existing database: {db_path}")

# ============================================================
# Model definitions
# ============================================================
MODELS = [
    {
        "name":      "GFDL-ESM4",
        "institute": "NOAA-GFDL",
        "ensemble":  "r1i1p1f1",
        "grid":      "gr",
        "path":      Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/"
                          "historical/r1i1p1f1/Omon/chl/gr/v20190726"),
        "pattern":   "chl_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_*.nc",
    },
    {
        "name":      "CESM2",
        "institute": "NCAR",
        "ensemble":  "r1i1p1f1",
        "grid":      "gn",
        "path":      Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/"
                          "historical/r1i1p1f1/Omon/chl/gn"),
        "pattern":   "chl_Omon_CESM2_historical_r1i1p1f1_gn_*.nc",
    },
    {
        "name":      "MPI-ESM1-2-LR",
        "institute": "MPI-M",
        "ensemble":  "r22i1p1f1",
        "grid":      "gn",
        "path":      Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/"
                          "historical/r22i1p1f1/Omon/chl/gn"),
        "pattern":   "chl_Omon_MPI-ESM1-2-LR_historical_r22i1p1f1_gn_*.nc",
    },
]

# ============================================================
# Basin definitions (lat_min, lat_max, lon_min, lon_max)
# lon in -180..180
# ============================================================
BASINS = {
    "Global":   (-90,  90, -180,  180),
    "Atlantic": (-70,  70,  -80,   20),
    "Pacific":  (-60,  65, -180,  -70),
    "Indian":   (-70,  30,   20,  120),
    "Southern": (-90, -40, -180,  180),
    "Arctic":   ( 65,  90, -180,  180),
}


# ============================================================
# Helper: load and preprocess a model cube
# ============================================================
def load_model_cube(model: dict) -> iris.cube.Cube | None:
    paths = sorted(model["path"].glob(model["pattern"]))
    if not paths:
        # Try without version subdir
        paths = sorted(model["path"].parent.glob(model["pattern"]))
    if not paths:
        logger.warning(f"No files found for {model['name']} in {model['path']}")
        return None

    logger.info(f"Loading {model['name']}: {len(paths)} files")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in paths])

    unify_time_units(cubes)
    equalise_attributes(cubes)
    for c in cubes:
        c.attributes = {}
    cubes = cubes.concatenate()

    if len(cubes) != 1:
        logger.warning(f"{model['name']}: expected 1 cube, got {len(cubes)}")

    cube = cubes[0]
    logger.info(f"{model['name']} original shape: {cube.shape}")

    # Surface level — index 0
    if cube.ndim == 4:
        cube = cube[:, 0, :, :]
    logger.info(f"{model['name']} after surface: {cube.shape}")

    # Time selection 1997-2014
    cube = cube.extract(Constraint(time=lambda c: 1997 <= c.point.year <= 2014))
    if cube is None:
        logger.warning(f"{model['name']}: no data for 1997-2014")
        return None
    logger.info(f"{model['name']} after time: {cube.shape}")
    return cube


# ============================================================
# Helper: mask cube to basin
# ============================================================
def mask_to_basin(cube: iris.cube.Cube, basin_name: str,
                  lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float) -> iris.cube.Cube:
    """Return cube masked to basin lat/lon bounds."""
    # Get lat/lon coords
    lat_coord = cube.coord("latitude")
    lon_coord = cube.coord("longitude")

    # Normalise longitudes to -180..180 if needed
    lons = lon_coord.points.copy()
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)

    lats = lat_coord.points

    # Build mask
    lat_ok = (lats >= lat_min) & (lats <= lat_max)
    if lon_min < lon_max:
        lon_ok = (lons >= lon_min) & (lons <= lon_max)
    else:  # wraps around (Pacific)
        lon_ok = (lons >= lon_min) | (lons <= lon_max)

    # For 2D lat/lon grids
    if lats.ndim == 2:
        mask2d = ~(lat_ok & lon_ok)
    else:
        mask2d = ~np.outer(lat_ok, lon_ok)

    # Apply mask to all time steps
    masked_data = np.ma.array(cube.data)
    if masked_data.ndim == 3:
        masked_data[:, mask2d] = np.ma.masked
    elif masked_data.ndim == 2:
        masked_data[mask2d] = np.ma.masked

    masked_cube = cube.copy(data=masked_data)
    masked_cube.rename(f"{cube.name()} ({basin_name})")
    return masked_cube


# ============================================================
# Helper: save processed cube to data_root_dir
# ============================================================
def save_cube(cube: iris.cube.Cube, model: dict,
              suffix: str = "") -> Path:
    fname = (f"chl_Omon_{model['name']}_historical_{model['ensemble']}_"
             f"{model['grid']}_199701-201412{suffix}.nc")
    out   = data_root_dir / fname
    iris.save(cube, str(out))
    return out


# ============================================================
# Build ClimateEval suite
# ============================================================
def get_suite() -> Suite:
    return Suite(
        suite_name,
        diagnostic_kwargs={
            "data_root_dir": data_root_dir,
            "fail_on_missing_data": False,
            "download_missing_data": False,
        },
        variable_kwargs={"timerange": "19970101/20141231"},
    )


# ============================================================
# Main — one get_database call per model, all to same db file
# ============================================================

all_cubes     = []
all_data_info = []

for model in MODELS:
    cube = load_model_cube(model)
    if cube is None:
        continue
    data_info = DataSourceInformation(
        name=model["name"],
        category="CMIP6",
        institute=model["institute"],
        exp="historical",
        variant=model["ensemble"],
        references=(
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP."
            f"{model['institute']}.{model['name']}",
        ),
    )
    all_cubes.append(cube)
    all_data_info.append(data_info)
    logger.success(f"Loaded {model['name']}: {cube.shape}")

if not all_cubes:
    raise RuntimeError("No model cubes loaded")

# ── Global: one call per model, same database ────────────────
for cube, data_info in zip(all_cubes, all_data_info):
    logger.info(f"Running ClimateEval — Global — {data_info.name}")
    suite = get_suite()
    suite.get_database(
        iris.cube.CubeList([cube]),
        data_info,
        database_resource=f"duckdb:///{db_path}",
    )

# ── Basins: one db per basin, all models in each ─────────────
for basin_name, (la0, la1, lo0, lo1) in BASINS.items():
    if basin_name == "Global":
        continue
    db_basin = Path.home() / f"{suite_name}_{basin_name}.ddb"
    if db_basin.is_file():
        db_basin.unlink()

    for cube, data_info in zip(all_cubes, all_data_info):
        logger.info(f"Running ClimateEval — {basin_name} — {data_info.name}")
        try:
            masked = mask_to_basin(cube, basin_name, la0, la1, lo0, lo1)
            suite_b = get_suite()
            suite_b.get_database(
                iris.cube.CubeList([masked]),
                data_info,
                database_resource=f"duckdb:///{db_basin}",
            )
        except Exception as e:
            logger.warning(f"{data_info.name} {basin_name} failed: {e}")

    logger.success(f"Basin {basin_name} database: {db_basin}")

logger.success(f"Global database: {db_path}")
logger.info("Generate HTML reports:")
logger.info(f"  python generate_html.py {db_path}")
for basin_name in BASINS:
    if basin_name != "Global":
        logger.info(f"  python generate_html.py ~/Tier2_ocean_{basin_name}.ddb")
