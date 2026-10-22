from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from esmvalcore.config import CFG
from esmvaltool.cmorizers.data.cmorizer import DATASETS_FILE, _Formatter

import climateeval.data
import climateeval.data._esmvaltool_cmorizer
from climateeval import Variable
from climateeval.data import DataSourceInformation, ESMValToolCMORizerDataSource

ESMVALTOOL_DATASETS: dict = yaml.safe_load(DATASETS_FILE.read_text())["datasets"]

DATA_SOURCES: list[type[ESMValToolCMORizerDataSource]] = sorted(
    (
        obj
        for obj in vars(climateeval.data).values()
        if isinstance(obj, type)
        and issubclass(obj, ESMValToolCMORizerDataSource)
        and obj is not ESMValToolCMORizerDataSource
    ),
    key=lambda cls: cls.__name__,
)

# ESMValTool ships a CMORizer for these datasets, but no downloader: the raw
# data has to be staged by hand before ``download()`` can format it.
NO_AUTOMATIC_DOWNLOAD = frozenset({"EN4"})

parametrize_data_sources = pytest.mark.parametrize(
    "data_source_cls",
    DATA_SOURCES,
    ids=[cls.__name__ for cls in DATA_SOURCES],
)


@parametrize_data_sources
def test_data_source_is_known_to_esmvaltool(
    data_source_cls: type[ESMValToolCMORizerDataSource],
) -> None:
    """``name`` must match an ESMValTool CMORizer dataset."""
    assert data_source_cls().name in ESMVALTOOL_DATASETS


@parametrize_data_sources
def test_data_source_tier_matches_esmvaltool(
    data_source_cls: type[ESMValToolCMORizerDataSource],
) -> None:
    """``_tier`` selects the output directory, so it must match ESMValTool."""
    name = data_source_cls().name
    assert data_source_cls._tier == ESMVALTOOL_DATASETS[name]["tier"]


@parametrize_data_sources
def test_data_source_can_be_downloaded_automatically(
    data_source_cls: type[ESMValToolCMORizerDataSource],
) -> None:
    """``download()`` relies on an ESMValTool downloader for the dataset."""
    name = data_source_cls().name
    if name in NO_AUTOMATIC_DOWNLOAD:
        pytest.skip(f"ESMValTool has no downloader for {name}")
    assert _Formatter.has_downloader(name)


# ----------------------------------------------------------------------------
# Generic download/format logic, with ESMValTool's ``DataCommand`` faked
# ----------------------------------------------------------------------------


class _DummyCMORizer(ESMValToolCMORizerDataSource):
    _information = DataSourceInformation(
        name="DUMMY-OBS",
        category="observation",
        institute="Nowhere",
    )
    _file_prefix_pattern: ClassVar[str] = "OBS_DUMMY-OBS_*"
    _supported_frequencies: ClassVar[tuple[str, ...] | None] = ("mon",)
    _supported_var_names: ClassVar[tuple[str, ...] | None] = ("tas", "pr")
    _tier: ClassVar[int] = 3


def _formatted_name(var_name: str, prefix: str = "OBS_DUMMY-OBS") -> str:
    return f"{prefix}_ground_1_Amon_{var_name}_200001-200012.nc"


class _FakeDataCommand:
    """Record calls and write CMORizer-like output into ``CFG["output_dir"]``."""

    calls: ClassVar[list[tuple[str, str, Path, Path]]] = []
    outputs: ClassVar[list[str]] = []

    def download(self, dataset: str, original_data_dir: Path) -> None:
        output_dir = Path(CFG["output_dir"])
        self.calls.append(("download", dataset, original_data_dir, output_dir))
        (original_data_dir / "raw_download.dat").write_text("raw")

    def format(self, dataset: str, original_data_dir: Path) -> None:
        output_dir = Path(CFG["output_dir"])
        self.calls.append(("format", dataset, original_data_dir, output_dir))
        assert (original_data_dir / "raw_download.dat").exists()
        # An older run that must be ignored, and the latest one.
        for stamp, names in (
            ("20000101_000000", ["stale"]),
            ("20990101_000000", self.outputs),
        ):
            out = output_dir / f"data_formatting_{stamp}" / "Tier3" / dataset
            out.mkdir(parents=True)
            for name in names:
                (out / name).write_text(f"{stamp} {name}")


@pytest.fixture
def fake_data_command(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDataCommand]:
    monkeypatch.setattr(_FakeDataCommand, "calls", [])
    monkeypatch.setattr(
        _FakeDataCommand,
        "outputs",
        [
            _formatted_name("tas"),
            _formatted_name("tas").replace("2000", "2001"),
            _formatted_name("pr"),
            _formatted_name("tas", prefix="OTHER_DUMMY-OBS"),  # filtered by prefix
            "notes.txt",
        ],
    )
    monkeypatch.setattr(
        climateeval.data._esmvaltool_cmorizer,
        "DataCommand",
        _FakeDataCommand,
    )
    return _FakeDataCommand


@pytest.fixture
def private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tmp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


def test_check_supported_variables() -> None:
    _DummyCMORizer()._check_supported_variables(Variable("tas", "tas", "mon"))


def test_check_unsupported_variable() -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'tos'"):
        _DummyCMORizer()._check_supported_variables(Variable("tos", "tos", "mon"))


def test_check_unsupported_frequency() -> None:
    msg = (
        "does not provide variable 'tas' with frequency 'day' "
        ".*supported frequencies: mon"
    )
    with pytest.raises(NotImplementedError, match=msg):
        _DummyCMORizer()._check_supported_variables(Variable("tas", "tas", "day"))


def test_check_no_restrictions_by_default() -> None:
    class _Unrestricted(_DummyCMORizer):
        _supported_frequencies = None
        _supported_var_names = None

    _Unrestricted()._check_supported_variables(Variable("tos", "tos", "day"))


def test_download_unsupported_variable_does_not_call_esmvaltool(
    tmp_path: Path,
    fake_data_command: type[_FakeDataCommand],
) -> None:
    with pytest.raises(NotImplementedError):
        _DummyCMORizer().download(Variable("tos", "tos", "mon"), tmp_path)
    assert fake_data_command.calls == []


@pytest.mark.usefixtures("private_tempdir")
def test_download_and_format(
    tmp_path: Path,
    fake_data_command: type[_FakeDataCommand],
) -> None:
    data_root_dir = tmp_path / "data"
    data_source = _DummyCMORizer()

    files = data_source.download(Variable("tas", "tas", "mon"), str(data_root_dir))

    # ESMValTool is asked to download into a temporary directory and then to
    # format from exactly that directory, each with its own output directory.
    (dl, dl_name, dl_dir, dl_out), (fmt, fmt_name, fmt_dir, fmt_out) = (
        fake_data_command.calls
    )
    assert (dl, fmt) == ("download", "format")
    assert dl_name == fmt_name == "DUMMY-OBS"
    assert dl_dir == dl_out == fmt_dir
    assert fmt_out != fmt_dir

    tas_dir = data_root_dir / "observation_DUMMY-OBS" / "mon" / "tas"
    pr_dir = data_root_dir / "observation_DUMMY-OBS" / "mon" / "pr"
    assert sorted(files) == sorted(
        [
            tas_dir / _formatted_name("tas"),
            tas_dir / _formatted_name("tas").replace("2000", "2001"),
            pr_dir / _formatted_name("pr"),
        ],
    )
    # Files come from the latest formatting run and were moved, not copied.
    for file in files:
        assert file.read_text() == f"20990101_000000 {file.name}"
    left_over = (fmt_out / "data_formatting_20990101_000000").rglob("*.nc")
    assert [p.name for p in left_over] == [
        _formatted_name("tas", prefix="OTHER_DUMMY-OBS"),
    ]
    assert sorted(p.name for p in tas_dir.iterdir()) == sorted(
        p.name for p in files if p.parent == tas_dir
    )


@pytest.mark.usefixtures("private_tempdir")
def test_download_does_not_overwrite_other_variables(
    tmp_path: Path,
    fake_data_command: type[_FakeDataCommand],
) -> None:
    data_root_dir = tmp_path / "data"
    base = data_root_dir / "observation_DUMMY-OBS" / "mon"
    existing_pr = base / "pr" / "existing_pr.nc"
    existing_pr.parent.mkdir(parents=True)
    existing_pr.write_text("old pr")
    existing_tas = base / "tas" / _formatted_name("tas")
    existing_tas.parent.mkdir(parents=True)
    existing_tas.write_text("old tas")

    files = _DummyCMORizer().download(Variable("tas", "tas", "mon"), data_root_dir)

    # The requested variable is (re)written, other present variables are kept.
    assert {f.parent.name for f in files} == {"tas"}
    assert existing_tas.read_text() == f"20990101_000000 {existing_tas.name}"
    assert sorted(p.name for p in (base / "pr").iterdir()) == ["existing_pr.nc"]
    assert existing_pr.read_text() == "old pr"


@pytest.mark.usefixtures("private_tempdir")
def test_download_nothing_formatted(
    tmp_path: Path,
    fake_data_command: type[_FakeDataCommand],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_data_command, "outputs", [])

    files = _DummyCMORizer().download(Variable("tas", "tas", "mon"), tmp_path)

    assert files == []
    assert not (tmp_path / "observation_DUMMY-OBS").exists()


def test_set_esmvaltool_output_dir(tmp_path: Path) -> None:
    original = CFG["output_dir"]

    with _DummyCMORizer()._set_esmvaltool_output_dir(tmp_path):
        assert Path(CFG["output_dir"]) == tmp_path

    assert CFG["output_dir"] == original


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Formatted files of extra variables are registered with Variable(), "
        "which raises for variables missing from variables.yml, so an "
        "opportunistically formatted extra variable aborts the download of "
        "the requested one"
    ),
)
@pytest.mark.usefixtures("private_tempdir")
def test_download_ignores_unknown_extra_variables(
    tmp_path: Path,
    fake_data_command: type[_FakeDataCommand],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fake_data_command,
        "outputs",
        [_formatted_name("tas"), _formatted_name("notavariable")],
    )

    files = _DummyCMORizer().download(Variable("tas", "tas", "mon"), tmp_path)

    assert [f.name for f in files] == [_formatted_name("tas")]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "download() creates its scratch dirs via "
        "Path(tempfile.TemporaryDirectory().name).mkdir(), which is never "
        "cleaned up: raw downloads and unused formatted files accumulate in "
        "the system temp dir"
    ),
)
def test_download_cleans_up_temporary_directories(
    tmp_path: Path,
    private_tempdir: Path,
    fake_data_command: type[_FakeDataCommand],
) -> None:
    _DummyCMORizer().download(Variable("tas", "tas", "mon"), tmp_path / "data")

    assert list(private_tempdir.iterdir()) == []
