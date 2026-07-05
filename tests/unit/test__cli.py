"""Tests for :mod:`climateeval._cli` (argument handling)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval._cli import _infer_model_name, main

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


def test_infer_model_name_from_cmor_filename(tmp_path: Path) -> None:
    fname = "tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc"
    (tmp_path / fname).touch()
    assert _infer_model_name(tmp_path) == "MPI-ESM1-2-LR"


def test_infer_model_name_from_single_file(tmp_path: Path) -> None:
    path = tmp_path / "pr_Amon_CESM2_historical_r1i1p1f1_gn_199001-200912.nc"
    path.touch()
    assert _infer_model_name(path) == "CESM2"


def test_infer_model_name_non_cmor_returns_none(tmp_path: Path) -> None:
    (tmp_path / "data.nc").touch()
    assert _infer_model_name(tmp_path) is None
