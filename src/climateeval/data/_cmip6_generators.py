"""Generate data sources from CMIP6 simulations."""

from __future__ import annotations

from typing import Any, ClassVar

from esmvalcore.cmor.table import CMOR_TABLES

from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSourceGenerator


class CMIP6DataSourceGenerator(ESMValToolIntakeDataSourceGenerator):
    """Generate data sources from CMIP6 output."""

    def _get_references(self, **facets: Any) -> tuple[str, ...]:
        """Get data source references.

        The ``exp`` facet may be a list of experiments (e.g. ``["historical",
        "ssp245"]`` for experiments that are concatenated into a single time
        series). In that case, one reference per activity is returned.
        """
        exps = facets["exp"]
        if isinstance(exps, str):
            exps = [exps]

        references: list[str] = []
        for exp in exps:
            activity = CMOR_TABLES["CMIP6"].activities[exp][0]
            reference = (
                f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.{activity}."
                f"{facets['institute']}.{facets['dataset']}"
            )
            if reference not in references:
                references.append(reference)

        return tuple(references)


class CMIP6Abrupt4xCO2R1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 abrupt-4xCO2 simulations (r1i1p1f1)."""

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "abrupt-4xCO2",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6PiControlR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 piControl simulations (r1i1p1f1)."""

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "piControl",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6HistoricalR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 historical simulations (r1i1p1f1)."""

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "historical",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
        "timerange": "19790101/20141231",
    }


class CMIP6HistoricalAllMembers(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 historical simulations (all members).

    Like :class:`CMIP6HistoricalR1I1P1F1`, but every ``r*i1p1f1`` member of
    every model is discovered as its own data source. The member ends up in
    :attr:`~climateeval.data.DataSourceInformation.variant`, so the data source
    IDs stay unique. Use this instead of the single-member generator when a
    per-model *ensemble* is needed, e.g. for probabilistic scores that are only
    defined for more than one member.
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "historical",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r*i1p1f1",
        "grid": "*",
        "timerange": "19790101/20141231",
    }


class CMIP6HistoricalSSP245(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 historical + SSP2-4.5 (all members).

    The ``historical`` and ``ssp245`` experiments are concatenated into a
    single time series (ESMValCore accepts a list of experiments in the ``exp``
    facet), which extends the CMIP6 reference beyond the end of the historical
    simulations in 2014. Every ``r*i1p1f1`` member of every model is discovered
    as its own data source, with the member in
    :attr:`~climateeval.data.DataSourceInformation.variant`.

    The end of the time range is a wildcard so that models whose SSP2-4.5
    simulation stops before 2100 are not discarded; ESMValCore replaces it with
    the last date that is actually available for each data source.
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": ["historical", "ssp245"],
        "dataset": "*",
        "institute": "*",
        "ensemble": "r*i1p1f1",
        "grid": "*",
        "timerange": "19790101/*",
    }


class CMIP6HistAerR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 hist-aer simulations (r1i1p1f1).

    ``hist-aer`` (DAMIP) is the historical simulation driven by anthropogenic
    aerosol forcing only. Together with ``historical`` it isolates the
    aerosol-forced component of the historical response.
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "hist-aer",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6AmipR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 amip simulations (r1i1p1f1).

    ``amip`` is the atmosphere-only simulation with prescribed observed sea
    surface temperatures and sea ice.
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "amip",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6Amip4xCO2R1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 amip-4xCO2 simulations (r1i1p1f1).

    ``amip-4xCO2`` (CFMIP) repeats the ``amip`` simulation with the CO2 seen by
    the radiation code quadrupled. Differenced against ``amip``, it gives the
    effective radiative forcing at fixed sea surface temperature.
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "amip-4xCO2",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6LgmR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 lgm simulations (r1i1p1f1).

    ``lgm`` (PMIP4) is the Last Glacial Maximum equilibrium simulation
    (21 ka BP).
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "lgm",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6MidHoloceneR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 midHolocene simulations (r1i1p1f1).

    ``midHolocene`` (PMIP4) is the mid-Holocene equilibrium simulation
    (6 ka BP).
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "midHolocene",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6Lig127kR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 lig127k simulations (r1i1p1f1).

    ``lig127k`` (PMIP4) is the Last Interglacial equilibrium simulation
    (127 ka BP).
    """

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "lig127k",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }
