"""ClimateEval benchmarking report generation and serving."""

from climateeval.report._html import build_html_shell, generate_portable_html
from climateeval.report._server import serve
from climateeval.report._static_figures import generate_static_plots

__all__ = [
    "build_html_shell",
    "generate_portable_html",
    "generate_static_plots",
    "serve",
]
