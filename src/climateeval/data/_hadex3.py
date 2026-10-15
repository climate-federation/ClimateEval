"""Provide HadEX3 data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class HadEX3(ESMValToolCMORizerDataSource):
    """Manage HadEX3 data.

    HadEX3 (v3.0.4) is a gridded data set of ETCCDI (Expert Team on Climate
    Change Detection and Indices) extremes indices derived from
    quality-controlled land station series. Grid boxes without nearby stations
    are masked, so the fields are land-only and spatially incomplete; this is
    the observational reference for evaluating simulated temperature and
    precipitation extremes.

    Four indices are provided: the annual maximum of daily maximum temperature
    (``txx``), the annual minimum of daily minimum temperature (``tnn``) and the
    annual maximum 1-day and 5-day precipitation totals (``rx1day`` and
    ``rx5day``).

    Notes
    -----
    The ESMValTool CMORizer writes every index twice, once as an annual field
    (MIP ``yr``) and once as a monthly field (MIP ``Amon``), under the same
    variable name. Formatted files are sorted into directories by variable name
    and by the frequency that was requested, so a mixture of both products would
    end up in the same directory. Only the annual indices (the canonical ETCCDI
    product) are therefore used here, and only the ``yr`` frequency is
    supported.

    """

    _information = DataSourceInformation(
        name="HadEX3",
        category="observation",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2019JD032263",),
    )

    # The CMORizer writes each index as an annual (``yr``) *and* a monthly
    # (``Amon``) field with the same variable name; only pick the annual one.
    _file_prefix_pattern = "*_yr_*"
    _supported_frequencies = ("yr",)
    _supported_var_names = ("rx1day", "rx5day", "tnn", "txx")
