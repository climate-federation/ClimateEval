"""Tests for :mod:`climateeval._cli` (argument handling)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from climateeval._cli import _infer_model_name, main
from tests.unit.test__check import _make_cube, _save, _suite

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


@pytest.mark.parametrize("command", ["benchmark", "check", "report"])
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


def test_check_missing_model_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Model path not found"):
        main(["check", str(tmp_path / "nope")])


def test_check_empty_model_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No NetCDF data found"):
        main(["check", str(tmp_path)])


def _write_check_fixture(tmp_path: Path, var_name: str) -> tuple[str, str]:
    """Save one synthetic cube and a single-variable suite; return CLI args."""
    data = tmp_path / "data"
    _save(_make_cube(var_name), data)
    suite = _suite(tmp_path, [{"id": "tas", "var_name": "tas", "frequency": "mon"}])
    return str(data), str(suite)


def test_check_passing_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, suite = _write_check_fixture(tmp_path, "tas")
    main(["check", data, "--suite", suite, "--timerange", "20000101/20011231"])
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "Summary:" in out


def test_check_failing_data_exits_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, suite = _write_check_fixture(tmp_path, "t2m")
    with pytest.raises(SystemExit) as excinfo:
        main(["check", data, "--suite", suite, "--timerange", "20000101/20011231"])
    assert excinfo.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, suite = _write_check_fixture(tmp_path, "tas")
    main(
        ["check", data, "--suite", suite, "--timerange", "20000101/20011231", "--json"],
    )
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True


def test_suites_list(capsys: pytest.CaptureFixture[str]) -> None:
    main(["suites", "list"])
    out = capsys.readouterr().out
    assert "Tier2_atmosphere_monthly" in out
    # The listing now expands each suite into its diagnostics: the diagnostic
    # name, its class and a variable count.
    assert "6 diagnostics" in out
    assert "annual_cycle" in out  # a diagnostic name
    assert "Map" in out  # a diagnostic class
    assert "vars" in out


def test_suites_list_filter(capsys: pytest.CaptureFixture[str]) -> None:
    main(["suites", "list", "Tier3"])
    out = capsys.readouterr().out
    assert "Tier3_dynamics" in out
    assert "Tier2_atmosphere_monthly" not in out


def test_suites_list_no_match(capsys: pytest.CaptureFixture[str]) -> None:
    main(["suites", "list", "does-not-exist"])
    assert "No suites match" in capsys.readouterr().out


def test_bare_suites_defaults_to_list(capsys: pytest.CaptureFixture[str]) -> None:
    main(["suites"])
    assert "Tier1_sanity_checks" in capsys.readouterr().out
