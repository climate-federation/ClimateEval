from benchmarking import COORDINATES, VARIABLES, Coordinate, Variable


def test_variables() -> None:
    assert VARIABLES


def test_coordinates() -> None:
    assert COORDINATES


def test_tas() -> None:
    var = Variable("tas")

    assert var.var_name == "tas"
    assert var.coordinates == [
        Coordinate("time"),
        Coordinate("lat"),
        Coordinate("lon"),
    ]
    assert var.realm == "atmosphere"
    assert var.standard_name == "air_temperature"
    assert var.units == "K"
