#!/usr/bin/env python
"""Entry point for the ClimateEval benchmarking report.

Default behaviour: start a local web server and open the browser.
Use --save-portable-html-output to also write a self-contained HTML file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from climateeval.report import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and serve a ClimateEval benchmarking report. "
            "By default a local web server is started and the browser is opened. "
            "Pass --save-portable-html-output to also write a self-contained HTML file."
        ),
    )
    parser.add_argument(
        "db",
        type=Path,
        nargs="+",
        help="Path(s) to DuckDB .ddb files.",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="NAME",
        help="Model name in the header.",
    )
    parser.add_argument(
        "--user",
        default="",
        metavar="NAME",
        help="Analyst name in the header.",
    )
    parser.add_argument(
        "--developer",
        default="",
        metavar="NAME",
        help="Developer / tool name.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    parser.add_argument(
        "--save-portable-html-output",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "Also save a self-contained HTML file. "
            "Omit PATH to place it next to the first database file."
        ),
    )
    args = parser.parse_args()

    db_paths = [p.resolve() for p in args.db]
    for p in db_paths:
        if not p.exists():
            parser.error(f"Database not found: {p}")

    save_html: Path | None = None
    if args.save_portable_html_output is not None:
        if args.save_portable_html_output:
            save_html = Path(args.save_portable_html_output)
        else:
            save_html = db_paths[0].with_name(db_paths[0].stem + "_report.html")

    serve(
        db_paths,
        model_name=args.model,
        user_name=args.user,
        developer=args.developer,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        save_html=save_html,
    )


if __name__ == "__main__":
    main()
