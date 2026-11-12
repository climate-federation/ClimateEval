from __future__ import annotations

import pytest

from climateeval.data import CERESEBAF, DataSourceInformation


@pytest.fixture
def data_source() -> CERESEBAF:
    return CERESEBAF()


def test_data_source(data_source: CERESEBAF) -> None:
    assert data_source.name == "CERES-EBAF"
    assert data_source.information == DataSourceInformation(
        name="CERES-EBAF",
        category="observation",
        institute="NASA-LaRC",
        references=("https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1",),
    )


# -- the cloud radiative effect: EBAF's own where staged -----------------------


def _flux_cube(var_name: str, value: float):  # noqa: ANN202
    import numpy as np
    from iris.coords import DimCoord
    from iris.cube import Cube

    time = DimCoord(
        np.array([15.5, 45.0, 74.5]),
        var_name="time",
        standard_name="time",
        units="days since 2001-01-01",
    )
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
    standard_names = {
        "rlut": "toa_outgoing_longwave_flux",
        "rlutcs": "toa_outgoing_longwave_flux_assuming_clear_sky",
        "rsut": "toa_outgoing_shortwave_flux",
        "rsutcs": "toa_outgoing_shortwave_flux_assuming_clear_sky",
    }
    return Cube(
        np.full((3, 2, 2), value, dtype=np.float32),
        var_name=var_name,
        standard_name=standard_names.get(var_name),
        units="W m-2",
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _stage_flux(root, data_source, var_name: str, value: float) -> None:  # noqa: ANN001
    import iris

    directory = root / data_source.id / "mon" / var_name
    directory.mkdir(parents=True)
    iris.save(
        _flux_cube(var_name, value),
        directory / f"obs4MIPs_CERES-EBAF-4-2-1_Amon_{var_name}_mon_20010116-20010316.nc",
    )


@pytest.mark.parametrize(
    ("var_name", "published", "all_sky", "clear_sky"),
    [("lwcre", "rltcre", "rlut", "rlutcs"), ("swcre", "rstcre", "rsut", "rsutcs")],
)
def test_cloud_radiative_effect_is_the_published_one_where_staged(
    data_source: CERESEBAF,
    tmp_path,  # noqa: ANN001
    var_name: str,
    published: str,
    all_sky: str,
    clear_sky: str,
) -> None:
    from climateeval import Variable

    # Clear-scene minus all-sky would give 28; EBAF's own effect is 26.
    _stage_flux(tmp_path, data_source, all_sky, 240.0)
    _stage_flux(tmp_path, data_source, clear_sky, 268.0)
    variable = Variable(var_name, var_name, "mon")

    derived = data_source.get_cube(tmp_path, variable, download_missing_data=False)
    assert derived.var_name == var_name
    assert float(derived.data.mean()) == pytest.approx(28.0)

    _stage_flux(tmp_path, data_source, published, 26.0)
    own = data_source.get_cube(tmp_path, variable, download_missing_data=False)
    assert own.var_name == var_name
    assert float(own.data.mean()) == pytest.approx(26.0)
    assert published in own.attributes["cloud_radiative_effect"]


def test_published_cre_does_not_touch_the_fluxes(data_source: CERESEBAF, tmp_path) -> None:  # noqa: ANN001
    from climateeval import Variable

    _stage_flux(tmp_path, data_source, "rlut", 240.0)
    _stage_flux(tmp_path, data_source, "rltcre", 26.0)
    cube = data_source.get_cube(
        tmp_path,
        Variable("rlut", "rlut", "mon"),
        download_missing_data=False,
    )
    assert float(cube.data.mean()) == pytest.approx(240.0)
