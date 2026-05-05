import numpy as np
from climateeval import NO_LEVEL, VerticalLevel


def test_no_level() -> None:
    assert isinstance(NO_LEVEL, VerticalLevel)
    assert NO_LEVEL.coordinate_name == ""
    assert NO_LEVEL.units == ""
    assert np.isnan(NO_LEVEL.value)
    assert repr(NO_LEVEL) == "NO_LEVEL"
