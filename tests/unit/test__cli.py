"""Tests for :mod:`climateeval._cli` (argument handling)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval._cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_no_command_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize("command", ["benchmark", "report"])
def test_subcommand_help(command: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0


def test_benchmark_missing_model_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Model path not found"):
        main(["benchmark", str(tmp_path / "nope"), "--name", "X"])


def test_benchmark_empty_model_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No NetCDF data found"):
        main(["benchmark", str(tmp_path), "--name", "X"])


def test_report_missing_database(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Database not found"):
        main(["report", str(tmp_path / "nope.ddb")])
