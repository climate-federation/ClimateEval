"""Tests for :mod:`climateeval._share` (publishing reports to DKRZ Swift).

The Swift REST API is never contacted: ``requests.get`` / ``requests.put`` are
replaced by a recorder, and ``~/.swiftenv`` is redirected into ``tmp_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import requests

from climateeval import _share

if TYPE_CHECKING:
    from pathlib import Path

_ENV_VARS = ("OS_STORAGE_URL", "OS_AUTH_TOKEN", "ST_USER", "ST_KEY", "ST_AUTH")


@dataclass
class _Response:
    headers: dict[str, str] = field(default_factory=dict)
    status: int = 200

    def raise_for_status(self) -> None:
        if self.status >= 400:
            msg = f"HTTP {self.status}"
            raise requests.HTTPError(msg)


@dataclass
class _Recorder:
    """Stand-in for the ``requests`` HTTP verbs used by the module."""

    get_response: _Response = field(default_factory=_Response)
    put_responses: list[_Response] = field(default_factory=list)
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append(("GET", url, kwargs))
        return self.get_response

    def put(self, url: str, **kwargs: Any) -> _Response:
        data = kwargs.get("data")
        if data is not None:  # the upload streams an open file: read it now
            kwargs = {**kwargs, "data": data.read()}
        self.calls.append(("PUT", url, kwargs))
        return self.put_responses.pop(0) if self.put_responses else _Response()


@pytest.fixture
def swiftenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the cached-token file into tmp_path and clear credential env vars."""
    path = tmp_path / "home" / ".swiftenv"
    monkeypatch.setattr(_share, "_SWIFTENV", path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return path


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(_share.requests, "get", recorder.get)
    monkeypatch.setattr(_share.requests, "put", recorder.put)
    return recorder


@pytest.fixture
def report(tmp_path: Path) -> Path:
    path = tmp_path / "my_report.html"
    path.write_text("<html>hi</html>", encoding="utf-8")
    return path


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- _parse_swiftenv -------------------------------------------------------


def test_parse_swiftenv_reads_setenv_lines(tmp_path: Path) -> None:
    path = _write(
        tmp_path / ".swiftenv",
        "# comment\n"
        "setenv OS_STORAGE_URL https://swift.dkrz.de/v1/dkrz_abc/\n"
        "setenv OS_AUTH_TOKEN AUTH_tk123\n"
        "setenv OTHER value\n"
        "export OS_AUTH_TOKEN ignored\n"
        "setenv\n",
    )
    assert _share._parse_swiftenv(path) == (
        "https://swift.dkrz.de/v1/dkrz_abc",
        "AUTH_tk123",
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "setenv OS_STORAGE_URL https://x\n",
        "setenv OS_AUTH_TOKEN tok\n",
        # Empty values are written by swift-token as a bare " ".
        'setenv OS_STORAGE_URL " "\nsetenv OS_AUTH_TOKEN tok\n',
        'setenv OS_STORAGE_URL https://x\nsetenv OS_AUTH_TOKEN " "\n',
    ],
)
def test_parse_swiftenv_incomplete_returns_none(tmp_path: Path, text: str) -> None:
    assert _share._parse_swiftenv(_write(tmp_path / ".swiftenv", text)) is None


# --- _tempauth -------------------------------------------------------------


def test_tempauth_sends_credentials_and_returns_endpoint(http: _Recorder) -> None:
    http.get_response = _Response(
        headers={"X-Storage-Url": "https://s/v1/acc/", "X-Auth-Token": "tok"},
    )
    assert _share._tempauth("https://auth", "acc:me", "pw", 5.0) == (
        "https://s/v1/acc",
        "tok",
    )
    assert http.calls == [
        (
            "GET",
            "https://auth",
            {"headers": {"X-Auth-User": "acc:me", "X-Auth-Key": "pw"}, "timeout": 5.0},
        ),
    ]


def test_tempauth_http_error_propagates(http: _Recorder) -> None:
    http.get_response = _Response(status=401)
    with pytest.raises(requests.HTTPError, match="401"):
        _share._tempauth("https://auth", "acc:me", "bad", 5.0)


# --- _resolve_endpoint -----------------------------------------------------


def test_resolve_prefers_environment_token(
    monkeypatch: pytest.MonkeyPatch,
    http: _Recorder,
    swiftenv: Path,
) -> None:
    _write(swiftenv, "setenv OS_STORAGE_URL https://cached\nsetenv OS_AUTH_TOKEN c\n")
    monkeypatch.setenv("OS_STORAGE_URL", "https://env/v1/acc/")
    monkeypatch.setenv("OS_AUTH_TOKEN", "envtok")
    monkeypatch.setenv("ST_USER", "acc:me")
    monkeypatch.setenv("ST_KEY", "pw")
    assert _share._resolve_endpoint(1.0) == ("https://env/v1/acc", "envtok")
    assert http.calls == []


def test_resolve_uses_cached_swiftenv(
    monkeypatch: pytest.MonkeyPatch,
    http: _Recorder,
    swiftenv: Path,
) -> None:
    _write(swiftenv, "setenv OS_STORAGE_URL https://cached/\nsetenv OS_AUTH_TOKEN c\n")
    # Only half of the env token is set, so it is not used.
    monkeypatch.setenv("OS_STORAGE_URL", "https://env")
    monkeypatch.setenv("ST_USER", "acc:me")
    monkeypatch.setenv("ST_KEY", "pw")
    assert _share._resolve_endpoint(1.0) == ("https://cached", "c")
    assert http.calls == []


def test_resolve_falls_back_to_tempauth_with_default_url(
    monkeypatch: pytest.MonkeyPatch,
    http: _Recorder,
    swiftenv: Path,
) -> None:
    _write(swiftenv, 'setenv OS_STORAGE_URL " "\nsetenv OS_AUTH_TOKEN " "\n')
    monkeypatch.setenv("ST_USER", "acc:me")
    monkeypatch.setenv("ST_KEY", "pw")
    http.get_response = _Response(
        headers={"X-Storage-Url": "https://s/v1/acc", "X-Auth-Token": "tok"},
    )
    assert _share._resolve_endpoint(7.0) == ("https://s/v1/acc", "tok")
    [(method, url, kwargs)] = http.calls
    assert (method, url, kwargs["timeout"]) == ("GET", _share.DEFAULT_AUTH_URL, 7.0)


@pytest.mark.usefixtures("swiftenv")
def test_resolve_tempauth_honours_st_auth(
    monkeypatch: pytest.MonkeyPatch,
    http: _Recorder,
) -> None:
    monkeypatch.setenv("ST_USER", "acc:me")
    monkeypatch.setenv("ST_KEY", "pw")
    monkeypatch.setenv("ST_AUTH", "https://other/auth/v1.0")
    http.get_response = _Response(
        headers={"X-Storage-Url": "https://s", "X-Auth-Token": "tok"},
    )
    _share._resolve_endpoint(1.0)
    assert http.calls[0][1] == "https://other/auth/v1.0"


@pytest.mark.usefixtures("swiftenv")
def test_resolve_without_credentials_raises(
    monkeypatch: pytest.MonkeyPatch,
    http: _Recorder,
) -> None:
    monkeypatch.setenv("ST_USER", "acc:me")  # no ST_KEY
    with pytest.raises(RuntimeError, match="No Swift credentials found"):
        _share._resolve_endpoint(1.0)
    assert http.calls == []


# --- share_file ------------------------------------------------------------


@pytest.fixture
def env_token(swiftenv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del swiftenv  # only needed to isolate the environment
    monkeypatch.setenv("OS_STORAGE_URL", "https://swift/v1/acc")
    monkeypatch.setenv("OS_AUTH_TOKEN", "tok")


@pytest.mark.usefixtures("env_token")
def test_share_file_creates_public_container_and_uploads(
    http: _Recorder,
    report: Path,
) -> None:
    url = _share.share_file(report, timeout=3.0)
    assert url == "https://swift/v1/acc/climateeval/my_report/index.html"
    assert http.calls == [
        (
            "PUT",
            "https://swift/v1/acc/climateeval",
            {
                "headers": {"X-Auth-Token": "tok", "X-Container-Read": ".r:*"},
                "timeout": 3.0,
            },
        ),
        (
            "PUT",
            url,
            {
                "data": b"<html>hi</html>",
                "headers": {"X-Auth-Token": "tok", "Content-Type": "text/html"},
                "timeout": None,
            },
        ),
    ]


@pytest.mark.usefixtures("env_token")
def test_share_file_private_custom_container_and_name(
    http: _Recorder,
    report: Path,
) -> None:
    url = _share.share_file(str(report), container="box", name="run1", public=False)
    assert url == "https://swift/v1/acc/box/run1/index.html"
    container_call, upload_call = http.calls
    assert container_call[1] == "https://swift/v1/acc/box"
    assert container_call[2]["headers"] == {"X-Auth-Token": "tok"}
    assert upload_call[1] == url


@pytest.mark.usefixtures("env_token")
def test_share_file_container_error_stops_before_upload(
    http: _Recorder,
    report: Path,
) -> None:
    http.put_responses = [_Response(status=403)]
    with pytest.raises(requests.HTTPError, match="403"):
        _share.share_file(report)
    assert len(http.calls) == 1


@pytest.mark.usefixtures("env_token")
def test_share_file_upload_error_propagates(http: _Recorder, report: Path) -> None:
    http.put_responses = [_Response(), _Response(status=500)]
    with pytest.raises(requests.HTTPError, match="500"):
        _share.share_file(report)
    assert len(http.calls) == 2


@pytest.mark.usefixtures("env_token")
def test_share_file_missing_file(http: _Recorder, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File to share not found"):
        _share.share_file(tmp_path / "missing.html")
    with pytest.raises(FileNotFoundError):
        _share.share_file(tmp_path)  # a directory is not a file
    assert http.calls == []
