from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING

import iris
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval import Variable
from climateeval.data import (
    CMORPH,
    GPCC,
    RAOBCORE,
    RICH,
    RSSMSU,
    UAHMSU,
    AMIPBoundaryConditions,
    BerkeleyEarthDaily,
    DataSourceInformation,
    GPCPDaily,
    HadGHCND,
    LocalObservationDataSource,
    RSSPrw,
)
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from pathlib import Path

# Class, data source ID, and the (variable, frequency) pairs it serves
_PRODUCTS = [
    (GPCPDaily, "observation_GPCP-1DD", [("pr", "day")]),
    (CMORPH, "observation_CMORPH-CDR", [("pr", "day")]),
    (
        BerkeleyEarthDaily,
        "observation_BerkeleyEarth-daily",
        [("tas", "day"), ("tasmax", "day"), ("tasmin", "day")],
    ),
    (HadGHCND, "observation_HadGHCND", [("tasmax", "day"), ("tasmin", "day")]),
    (RSSPrw, "observation_RSS-TPW", [("prw", "mon")]),
    (RSSMSU, "observation_RSS-MSU", [("tlt", "mon"), ("tmt", "mon")]),
    (UAHMSU, "observation_UAH-MSU", [("tlt", "mon"), ("tmt", "mon")]),
    (RICH, "observation_RICH", [("ta", "mon")]),
    (RAOBCORE, "observation_RAOBCORE", [("ta", "mon")]),
    (
        AMIPBoundaryConditions,
        "observation_AMIPBoundaryConditions",
        [("tos", "mon"), ("siconc", "mon")],
    ),
]
_CASES = [
    (cls, source_id, var_name, frequency)
    for cls, source_id, variables in _PRODUCTS
    for var_name, frequency in variables
]
_IDS = [f"{source_id}-{var_name}" for _, source_id, var_name, _ in _CASES]


def _make_cube(variable: Variable, year: int, value: float) -> Cube:
    """Create a small CMOR-like cube with 12 time steps within ``year``."""
    if variable.frequency == "day":
        points = np.arange(12) + 0.5
        units = f"days since {year}-03-01"
    else:
        # Mid-month points on the standard calendar (365 days)
        points = np.array(
            [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5],
        )
        units = f"days since {year}-01-01"
    time = DimCoord(points, var_name="time", standard_name="time", units=units)
    time.guess_bounds()
    lat = DimCoord(
        [-30.0, 30.0],
        bounds=[[-60.0, 0.0], [0.0, 60.0]],
        var_name="lat",
        standard_name="latitude",
        units="degrees_north",
    )
    lon = DimCoord(
        [45.0, 135.0],
        bounds=[[0.0, 90.0], [90.0, 180.0]],
        var_name="lon",
        standard_name="longitude",
        units="degrees_east",
    )
    dim_coords = [(time, 0)]
    shape: tuple[int, ...] = (12, 2, 2)
    if variable.ndim == 4:
        plev = DimCoord(
            [100000.0, 85000.0],
            var_name="plev",
            standard_name="air_pressure",
            units="Pa",
            attributes={"positive": "down"},
        )
        dim_coords += [(plev, 1), (lat, 2), (lon, 3)]
        shape = (12, 2, 2, 2)
    else:
        dim_coords += [(lat, 1), (lon, 2)]
    return Cube(
        np.full(shape, value, dtype=np.float32),
        var_name=variable.var_name,
        standard_name=variable.standard_name or None,
        long_name=variable.long_name,
        units=variable.units,
        dim_coords_and_dims=dim_coords,
    )


def _stage(root: Path, source_id: str, variable: Variable) -> None:
    """Stage one file per year (2000, 2001) as the observation bundle does."""
    data_dir = root / source_id / variable.frequency / variable.var_name
    data_dir.mkdir(parents=True)
    for year, value in ((2000, 1.0), (2001, 2.0)):
        iris.save(
            _make_cube(variable, year, value),
            data_dir / f"{variable.var_name}_Amon_test_v1_{year}0101-{year}1231.nc",
        )


@pytest.mark.parametrize(
    ("cls", "source_id", "var_name", "frequency"),
    _CASES,
    ids=_IDS,
)
def test_get_cube(
    tmp_path: Path,
    cls: type[LocalObservationDataSource],
    source_id: str,
    var_name: str,
    frequency: str,
) -> None:
    variable = Variable(var_name, var_name, frequency, timerange="20010101/20011231")
    _stage(tmp_path, source_id, variable)
    data_source = cls()

    cube = data_source.get_cube(tmp_path, variable, download_missing_data=False)

    assert data_source.id == source_id
    assert data_source.information.category == "observation"
    assert cube.var_name == var_name
    # Only the file of the requested year contributes
    time = cube.coord("time")
    assert {d.year for d in time.units.num2date(time.points)} == {2001}
    np.testing.assert_allclose(cube.data, 2.0)


@pytest.mark.parametrize(
    ("cls", "source_id", "var_name", "frequency"),
    _CASES,
    ids=_IDS,
)
def test_get_cube_all_years(
    tmp_path: Path,
    cls: type[LocalObservationDataSource],
    source_id: str,
    var_name: str,
    frequency: str,
) -> None:
    variable = Variable(var_name, var_name, frequency)
    _stage(tmp_path, source_id, variable)

    cube = cls().get_cube(tmp_path, variable)

    assert cube.shape[0] == 24


@pytest.mark.parametrize(
    ("cls", "source_id", "var_name", "frequency"),
    _CASES,
    ids=_IDS,
)
def test_missing_data(
    tmp_path: Path,
    cls: type[LocalObservationDataSource],
    source_id: str,
    var_name: str,
    frequency: str,
) -> None:
    variable = Variable(var_name, var_name, frequency)
    (tmp_path / source_id).mkdir()  # empty tree

    with pytest.raises(MissingDataError, match="locally staged observation bundle"):
        cls().get_cube(tmp_path, variable)
    with pytest.raises(MissingDataError, match=source_id):
        cls().get_cube(tmp_path, variable, download_missing_data=False)


def test_missing_data_names_prep_script(tmp_path: Path) -> None:
    variable = Variable("pr", "pr", "day")

    with pytest.raises(MissingDataError, match=r"obs/prep/gpcp_daily\.py"):
        GPCPDaily().get_cube(tmp_path, variable)


@pytest.mark.parametrize(
    ("cls", "var_name", "frequency", "message"),
    [
        (GPCPDaily, "tas", "day", "does not provide variable 'tas'"),
        (GPCPDaily, "pr", "mon", "with frequency 'mon'"),
        (RSSMSU, "ta", "mon", "does not provide variable 'ta'"),
    ],
)
def test_unsupported_variable(
    tmp_path: Path,
    cls: type[LocalObservationDataSource],
    var_name: str,
    frequency: str,
    message: str,
) -> None:
    variable = Variable(var_name, var_name, frequency)

    with pytest.raises(NotImplementedError, match=message):
        cls().get_cube(tmp_path, variable)


def test_download_not_possible(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="locally staged observation bundle"):
        GPCPDaily().download(Variable("pr", "pr", "day"), tmp_path)


def test_amip_boundary_conditions_configurable_name(tmp_path: Path) -> None:
    data_source_cls = AMIPBoundaryConditions.for_dataset(
        "PCMDI-AMIP-1-1-9",
        institute="PCMDI",
        references=("https://doi.org/10.22033/ESGF/input4MIPs.16911",),
    )
    variable = Variable("tos", "tos", "mon")
    _stage(tmp_path, "observation_PCMDI-AMIP-1-1-9", variable)

    data_source = data_source_cls()
    cube = data_source.get_cube(tmp_path, variable)

    assert issubclass(data_source_cls, AMIPBoundaryConditions)
    assert data_source.information == DataSourceInformation(
        name="PCMDI-AMIP-1-1-9",
        category="observation",
        institute="PCMDI",
        references=("https://doi.org/10.22033/ESGF/input4MIPs.16911",),
    )
    assert cube.var_name == "tos"


def test_ids_are_unique() -> None:
    ids = [source_id for _, source_id, _ in _PRODUCTS]
    assert len(set(ids)) == len(ids)


def test_gpcc_uses_esmvaltool_cmorizer(tmp_path: Path) -> None:
    """GPCC is served from the ESMValTool CMORizer output (staged or CMORized)."""
    variable = Variable("pr", "pr", "mon", timerange="20010101/20011231")
    _stage(tmp_path, "observation_GPCC", variable)
    data_source = GPCC()

    cube = data_source.get_cube(tmp_path, variable, download_missing_data=False)

    assert data_source.id == "observation_GPCC"
    assert data_source.information.references == (
        "https://doi.org/10.5676/DWD_GPCC/FD_M_V2018_100",
    )
    assert cube.var_name == "pr"
    np.testing.assert_allclose(cube.data, 2.0)


def test_gpcc_only_unmasked_one_degree_files() -> None:
    pattern = f"{GPCC._file_prefix_pattern}.nc"
    assert fnmatch("OBS_GPCC_reanaly_v2018-10_Amon_pr_189101-201612.nc", pattern)
    assert not fnmatch(
        "OBS_GPCC_reanaly_v2018-10-numgauge1_Amon_pr_189101-201612.nc",
        pattern,
    )
    assert not fnmatch("OBS_GPCC_reanaly_v2018-25_Amon_pr_189101-201612.nc", pattern)


def test_gpcc_download_missing_not_allowed(tmp_path: Path) -> None:
    variable = Variable("pr", "pr", "mon")

    with pytest.raises(DownloadError, match="downloads are disabled"):
        GPCC().get_cube(tmp_path, variable, download_missing_data=False)
