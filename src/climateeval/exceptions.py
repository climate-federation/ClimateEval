"""Provide ClimateEval exceptions and warnings."""


class ClimateEvalError(Exception):
    """General ClimateEval error."""


class CoordinateError(ClimateEvalError):
    """Coordinate error."""


class DownloadError(ClimateEvalError):
    """Download error."""


class MissingDataError(ClimateEvalError):
    """Missing data error."""
