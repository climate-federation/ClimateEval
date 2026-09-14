"""Generate data sources from CMIP6 piControl simulations."""

from __future__ import annotations

from typing import Any, ClassVar

from esmvalcore.cmor.table import CMOR_TABLES

from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSourceGenerator


class CMIP6DataSourceGenerator(ESMValToolIntakeDataSourceGenerator):
    """Generate data sources from CMIP6 output."""

    def _get_references(self, **facets: Any) -> tuple[str, ...]:
        """Get data source references."""
        activity = CMOR_TABLES["CMIP6"].activities[facets["exp"]][0]
        return (
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.{activity}."
            f"{facets['institute']}.{facets['dataset']}",
        )


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
