"""Publish files to DKRZ Swift object storage and return a shareable URL.

DKRZ operates an OpenStack Swift object store (https://swift.dkrz.de). This
module uploads a local file (typically a portable ``report.html``) into a
public-read container so it can be opened by anyone with the link, with no
DKRZ account required. It mirrors how ICONEval publishes ESMValTool output, so
ClimateEval reports land next to them with the same style of URL:

    https://swift.dkrz.de/v1/<account>/<container>/<name>/index.html

It talks to Swift's RESTful API directly with ``requests`` (no
``python-swiftclient`` dependency).

Authentication (first source that is available wins)
----------------------------------------------------
1. A pre-fetched token in the environment: ``OS_STORAGE_URL`` +
   ``OS_AUTH_TOKEN``.
2. A cached token in ``~/.swiftenv`` (the file written by DKRZ's ``swift-token``
   tool / ICONEval, in ``setenv OS_STORAGE_URL ...`` / ``setenv OS_AUTH_TOKEN
   ...`` form). This is the usual path on Levante and lets a report be
   published into the same **project** account (10 TiB, shared with
   colleagues) as ICONEval.
3. Legacy tempauth: ``ST_USER`` (``<account>:<user>``) + ``ST_KEY`` (password),
   authenticated once against ``ST_AUTH`` (defaults to the DKRZ endpoint). Use
   a project account for ``<account>`` to publish into the shared space.

To create/refresh a cached token, run ``swift-token`` on Levante (from
``py-python-swiftclient``) and enter your project account, username and
password.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from loguru import logger

#: Default DKRZ Swift tempauth endpoint (overridable via ``ST_AUTH``).
DEFAULT_AUTH_URL = "https://swift.dkrz.de/auth/v1.0"

#: Container that shared reports land in unless overridden.
DEFAULT_CONTAINER = "climateeval"

#: ACL that makes a container world-readable (anyone with the link can GET).
_PUBLIC_ACL = ".r:*"

#: Cached-token file written by DKRZ's ``swift-token`` tool / ICONEval.
_SWIFTENV = Path.home() / ".swiftenv"

#: Minimum fields in a ``setenv KEY VALUE`` line.
_SETENV_PARTS = 3


def _parse_swiftenv(path: Path) -> tuple[str, str] | None:
    """Return ``(storage_url, token)`` from a ``~/.swiftenv`` file, or None.

    The file uses tcsh ``setenv KEY VALUE`` lines; empty values are written as
    a bare ``" "`` and are ignored here.
    """
    storage_url = token = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= _SETENV_PARTS and parts[0] == "setenv":
            if parts[1] == "OS_STORAGE_URL":
                storage_url = parts[2]
            elif parts[1] == "OS_AUTH_TOKEN":
                token = parts[2]
    if storage_url and token and storage_url != '"' and token != '"':  # noqa: S105
        return storage_url.rstrip("/"), token
    return None


def _tempauth(auth_url: str, user: str, key: str, timeout: float) -> tuple[str, str]:
    """Authenticate via legacy tempauth and return ``(storage_url, token)``."""
    logger.info(f"Authenticating with Swift at {auth_url}")
    resp = requests.get(
        auth_url,
        headers={"X-Auth-User": user, "X-Auth-Key": key},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.headers["X-Storage-Url"].rstrip("/"), resp.headers["X-Auth-Token"]


def _resolve_endpoint(timeout: float) -> tuple[str, str]:
    """Return ``(storage_url, token)`` from env, ``~/.swiftenv`` or tempauth."""
    # 1. Explicit pre-fetched token in the environment.
    env_url = os.environ.get("OS_STORAGE_URL")
    env_token = os.environ.get("OS_AUTH_TOKEN")
    if env_url and env_token:
        return env_url.rstrip("/"), env_token

    # 2. Cached token from ~/.swiftenv.
    if _SWIFTENV.is_file():
        cached = _parse_swiftenv(_SWIFTENV)
        if cached is not None:
            logger.info(f"Using cached Swift token from {_SWIFTENV}")
            return cached

    # 3. Legacy tempauth via ST_USER / ST_KEY.
    user = os.environ.get("ST_USER")
    key = os.environ.get("ST_KEY")
    if user and key:
        auth_url = os.environ.get("ST_AUTH", DEFAULT_AUTH_URL)
        return _tempauth(auth_url, user, key, timeout)

    msg = (
        "No Swift credentials found. Use one of:\n"
        "  1. A cached token: run `swift-token` on Levante (enter your project "
        "account, username, password) to write ~/.swiftenv, then retry.\n"
        "  2. export OS_STORAGE_URL=<url> OS_AUTH_TOKEN=<token>\n"
        "  3. export ST_USER=<account>:<user> ST_KEY=<password>  "
        "(use a project account to publish into the shared space)."
    )
    raise RuntimeError(msg)


def share_file(
    path: str | Path,
    *,
    container: str = DEFAULT_CONTAINER,
    name: str | None = None,
    public: bool = True,
    timeout: float = 30.0,
) -> str:
    """Upload ``path`` to DKRZ Swift and return the public report URL.

    The file is stored as ``<container>/<name>/index.html`` so each report gets
    a clean, collision-free URL under its own prefix (matching ICONEval).

    Parameters
    ----------
    path
        Local file to upload (a self-contained HTML report).
    container
        Swift container to upload into; created if missing.
    name
        Prefix (pseudo-directory) for the report; defaults to the file stem.
    public
        If true (default), set a public-read ACL on the container so anyone
        with the returned URL can open the report without a DKRZ account.
    timeout
        Per-request timeout in seconds for auth and container calls (the
        upload itself is not time-limited).

    Returns
    -------
    str
        The URL of the published ``index.html``.
    """
    path = Path(path)
    if not path.is_file():
        msg = f"File to share not found: {path}"
        raise FileNotFoundError(msg)
    name = name or path.stem
    object_name = f"{name}/index.html"

    storage_url, token = _resolve_endpoint(timeout)
    headers = {"X-Auth-Token": token}

    # Ensure the container exists and is public-read (idempotent).
    put_headers = dict(headers)
    if public:
        put_headers["X-Container-Read"] = _PUBLIC_ACL
    resp = requests.put(
        f"{storage_url}/{container}",
        headers=put_headers,
        timeout=timeout,
    )
    resp.raise_for_status()

    # Upload the report (streamed from disk; no timeout for large files).
    object_url = f"{storage_url}/{container}/{object_name}"
    size_mb = path.stat().st_size / 1e6
    logger.info(f"Uploading {path.name} ({size_mb:.1f} MB) to {object_name}")
    with path.open("rb") as fh:
        resp = requests.put(
            object_url,
            data=fh,
            headers={**headers, "Content-Type": "text/html"},
            timeout=None,  # noqa: S113 - large uploads must not time out
        )
    resp.raise_for_status()

    return object_url
