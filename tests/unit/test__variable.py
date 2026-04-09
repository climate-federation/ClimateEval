import re

import pytest

from benchmarking import COORDINATES, VARIABLES, Coordinate, Variable


def test_variables() -> None:
    assert VARIABLES


def test_coordinates() -> None:
    assert COORDINATES


def test_tas() -> None:
    var = Variable("my_tas", "tas", "mon")

    assert var.id == "my_tas"
    assert var.var_name == "tas"
    assert var.frequency == "mon"
    assert var.level == "*"
    assert var.min_lat == -90.0
    assert var.max_lat == 90.0
    assert var.min_lon == 0.0
    assert var.max_lon == 360.0
    assert var.timerange == "*"
    assert var.coordinates == [
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    ]
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
    assert var.level == "*"
    assert var.min_lat == 0.0
    assert var.max_lat == 90.0
    assert var.min_lon == 0.0
    assert var.max_lon == 180.0
    assert var.timerange == "*"
    assert var.coordinates == [
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    ]
    assert var.derived is True
    assert var.long_name == "TOA Longwave Cloud Radiative Effect"
    assert var.realm == "atmosphere"
    assert var.standard_name == ""
    assert var.units == "W m-2"


def test_get_settings() -> None:
    var = Variable("my_lwcre", "lwcre", "yr", min_lat=0.0, max_lon=180.0)
    settings = var.get_settings()
    assert settings == {
        "level": "*",
        "min_lat": 0.0,
        "max_lat": 90.0,
        "min_lon": 0.0,
        "max_lon": 180.0,
        "timerange": "*",
    }


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
    msg = "Variable 'var' not supported yet"
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        Variable("my_var", "var", "mon")
