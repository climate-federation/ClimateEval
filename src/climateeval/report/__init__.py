"""ClimateEval benchmarking report generation and serving."""

from climateeval.report._html import build_html_shell, generate_portable_html
from climateeval.report._server import serve

__all__ = ["build_html_shell", "generate_portable_html", "serve"]
