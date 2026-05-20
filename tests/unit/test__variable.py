import re
from typing import Any

import pytest

from climateeval import (
    NO_LEVEL,
    Coordinate,
    Variable,
)
from climateeval._variable import COORDINATES, VARIABLES


def test_variables() -> None:
    assert VARIABLES


def test_coordinates() -> None:
    assert COORDINATES


def test_lat() -> None:
    coord = Coordinate("lat")

    assert coord.var_name == "lat"
    assert coord.long_name == "Latitude"
    assert coord.standard_name == "latitude"
    assert coord.units == "degrees_north"
    assert coord.allowed_coordinates == ()


def test_alevel() -> None:
    coord = Coordinate("alevel")

    assert coord.var_name == "alevel"
    assert coord.long_name == ""
    assert coord.standard_name == ""
    assert coord.units == ""
    assert coord.allowed_coordinates == (Coordinate("alt"), Coordinate("plev"))


@pytest.mark.parametrize(
    ("coord", "name"),
    [
        ("alt", "altitude"),
        ("hour", "hour"),
        ("plev", "air_pressure"),
    ],
)
def test_coordinate_name(coord: str, name: str) -> None:
    assert Coordinate(coord).name() == name


def test_tas() -> None:
    var = Variable("my_tas", "tas", "mon")

    assert var.id == "my_tas"
    assert var.var_name == "tas"
    assert var.frequency == "mon"

    settings = {
        "vertical_level": NO_LEVEL,
        "min_lat": -90.0,
        "max_lat": 90.0,
        "min_lon": 0.0,
        "max_lon": 360.0,
        "landsea_mask": "*",
        "ar6_regions_mask": (),
        "timerange": "*",
        "units": "K",
        "anomaly": "",
    }
    assert var.settings == settings
    for key, val in settings.items():
        assert getattr(var, key) == val

    assert var.coordinates == (
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    )
    assert var.derived is False
    assert var.long_name == "Near-Surface Air Temperature"
    assert var.realm == "atmos"
    assert var.standard_name == "air_temperature"
    assert var.cmip6_table_id == "Amon"
    assert var.ndim == 3


def test_ta() -> None:
    var = Variable("my_ta", "ta", "mon")

    assert var.id == "my_ta"
    assert var.var_name == "ta"
    assert var.frequency == "mon"

    settings = {
        "vertical_level": NO_LEVEL,
        "min_lat": -90.0,
        "max_lat": 90.0,
        "min_lon": 0.0,
        "max_lon": 360.0,
        "landsea_mask": "*",
        "ar6_regions_mask": (),
        "timerange": "*",
        "units": "K",
        "anomaly": "",
    }
    assert var.settings == settings
    for key, val in settings.items():
        assert getattr(var, key) == val

    assert var.coordinates == (
        Coordinate("time"),
        Coordinate("alevel"),
        Coordinate("lat"),
        Coordinate("lon"),
    )
    assert var.derived is False
    assert var.long_name == "Air Temperature"
    assert var.realm == "atmos"
    assert var.standard_name == "air_temperature"
    assert var.cmip6_table_id == "Amon"
    assert var.ndim == 4


def test_lwcre() -> None:
    var = Variable("my_lwcre", "lwcre", "mon", min_lat=0.0, max_lon=180.0)

    assert var.id == "my_lwcre"
    assert var.var_name == "lwcre"
    assert var.frequency == "mon"

    settings = {
        "vertical_level": NO_LEVEL,
        "min_lat": 0.0,
        "max_lat": 90.0,
        "min_lon": 0.0,
        "max_lon": 180.0,
        "landsea_mask": "*",
        "ar6_regions_mask": (),
        "timerange": "*",
        "units": "W m-2",
        "anomaly": "",
    }
    assert var.settings == settings
    for key, val in settings.items():
        assert getattr(var, key) == val

    assert var.coordinates == (
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    )
    assert var.derived is True
    assert var.long_name == "TOA Longwave Cloud Radiative Effect"
    assert var.realm == "atmos"
    assert var.standard_name == ""
    assert var.cmip6_table_id == "Amon"
    assert var.ndim == 3


@pytest.mark.parametrize(
    ("variable_kwargs", "expected_settings"),
    [
        (
            {"min_lat": 0.0, "max_lon": 180.0},
            {
                "vertical_level": NO_LEVEL,
                "min_lat": 0.0,
                "max_lat": 90.0,
                "min_lon": 0.0,
                "max_lon": 180.0,
                "landsea_mask": "*",
                "ar6_regions_mask": (),
                "timerange": "*",
                "units": "W m-2",
                "anomaly": "",
            },
        ),
        (
            {
                "units": "kW w-2",
                "landsea_mask": "land-only",
                "ar6_regions_mask": ("Caribbean", "E.Europe"),
            },
            {
                "vertical_level": NO_LEVEL,
                "min_lat": -90.0,
                "max_lat": 90.0,
                "min_lon": 0.0,
                "max_lon": 360.0,
                "landsea_mask": "land-only",
                "ar6_regions_mask": ("Caribbean", "E.Europe"),
                "timerange": "*",
                "units": "kW w-2",
                "anomaly": "",
            },
        ),
    ],
)
def test_settings(
    variable_kwargs: dict[str, Any],
    expected_settings: dict[str, Any],
) -> None:
    var = Variable("my_lwcre", "lwcre", "mon", **variable_kwargs)
    assert var.settings == expected_settings


def test_get_required_variables_derived() -> None:
    var = Variable("my_lwcre", "lwcre", "mon", min_lat=0.0, max_lon=180.0)
    required_vars = var.get_required_variables()
    assert required_vars == (
        Variable("rlut", "rlut", "mon", min_lat=0.0, max_lon=180.0),
        Variable("rlutcs", "rlutcs", "mon", min_lat=0.0, max_lon=180.0),
    )


def test_get_required_variables_not_derived() -> None:
    var = Variable("my_tas", "tas", "mon")
    required_vars = var.get_required_variables()
    assert required_vars == (var,)


@pytest.mark.parametrize(
    ("var_name", "frequency", "cmip6_table_id"),
    [
        ("pr", "yr", ""),
        ("pr", "mon", "Amon"),
        ("pr", "day", "day"),
        ("pr", "3hr", "3hr"),
        ("pr", "1hr", "E1hr"),
        ("tas", "yr", ""),
        ("tas", "mon", "Amon"),
        ("tas", "day", "day"),
        ("tas", "3hr", "3hr"),
        ("tas", "1hr", "AERhr"),
        ("tos", "yr", "Oyr"),
        ("tos", "mon", "Omon"),
        ("tos", "day", "Oday"),
        ("tos", "3hr", ""),
        ("tos", "1hr", ""),
    ],
)
def test_cmip6_table_id(var_name: str, frequency: str, cmip6_table_id: str) -> None:
    var = Variable(var_name, var_name, frequency)
    assert var.cmip6_table_id == cmip6_table_id


def test_var_not_implemented_fail() -> None:
    msg = r"Variable 'var' not supported yet"
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        Variable("my_var", "var", "mon")
