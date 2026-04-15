import re
from typing import Any

import pytest

from benchmarking import (
    COORDINATES,
    NO_LEVEL,
    VARIABLES,
    Coordinate,
    Variable,
)


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
    assert var.vertical_level is NO_LEVEL
    assert var.min_lat == -90.0
    assert var.max_lat == 90.0
    assert var.min_lon == 0.0
    assert var.max_lon == 360.0
    assert var.landsea_mask == "*"
    assert var.ar6_regions_mask == ()
    assert var.timerange == "*"
    assert var.coordinates == (
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    )
    assert var.derived is False
    assert var.long_name == "Near-Surface Air Temperature"
    assert var.realm == "atmosphere"
    assert var.standard_name == "air_temperature"
    assert var.units == "K"


def test_lwcre() -> None:
    var = Variable("my_lwcre", "lwcre", "yr", min_lat=0.0, max_lon=180.0)

    assert var.id == "my_lwcre"
    assert var.var_name == "lwcre"
    assert var.frequency == "yr"
    assert var.vertical_level is NO_LEVEL
    assert var.min_lat == 0.0
    assert var.max_lat == 90.0
    assert var.min_lon == 0.0
    assert var.max_lon == 180.0
    assert var.landsea_mask == "*"
    assert var.ar6_regions_mask == ()
    assert var.timerange == "*"
    assert var.coordinates == (
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    )
    assert var.derived is True
    assert var.long_name == "TOA Longwave Cloud Radiative Effect"
    assert var.realm == "atmosphere"
    assert var.standard_name == ""
    assert var.units == "W m-2"


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
            },
        ),
    ],
)
def test_settings(
    variable_kwargs: dict[str, Any],
    expected_settings: dict[str, Any],
) -> None:
    var = Variable("my_lwcre", "lwcre", "yr", **variable_kwargs)
    assert var.settings == expected_settings


def test_get_required_variables_derived() -> None:
    var = Variable("my_lwcre", "lwcre", "yr", min_lat=0.0, max_lon=180.0)
    required_vars = var.get_required_variables()
    assert required_vars == (
        Variable("rlut", "rlut", "yr", min_lat=0.0, max_lon=180.0),
        Variable("rlutcs", "rlutcs", "yr", min_lat=0.0, max_lon=180.0),
    )


def test_get_required_variables_not_derived() -> None:
    var = Variable("my_tas", "tas", "mon")
    required_vars = var.get_required_variables()
    assert required_vars == (var,)


def test_var_not_implemented_fail() -> None:
    msg = r"Variable 'var' not supported yet"
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        Variable("my_var", "var", "mon")
