from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import requests

from moda.LBC.nest_dales_in_HARMONIE import knmi_api


def _response(
    status_code: int,
    chunks,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status_code
    response.headers = headers or {}
    response.iter_content.return_value = chunks
    return response


def test_download_resumes_after_connection_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = tmp_path / "forecast.tar"
    interrupted_response = _response(
        200,
        _interrupted_chunks(),
        {"Content-Length": "6"},
    )
    resumed_response = _response(
        206,
        [b"def"],
        {"Content-Range": "bytes 3-5/6", "Content-Length": "3"},
    )
    get = MagicMock(side_effect=[interrupted_response, resumed_response])
    monkeypatch.setattr(knmi_api.requests, "get", get)
    monkeypatch.setattr(knmi_api.time, "sleep", lambda _: None)

    knmi_api._download_file("https://example.test/file", target_path)

    assert target_path.read_bytes() == b"abcdef"
    assert get.call_args_list == [
        call(
            "https://example.test/file",
            headers={},
            stream=True,
            timeout=60,
        ),
        call(
            "https://example.test/file",
            headers={"Range": "bytes=3-"},
            stream=True,
            timeout=60,
        ),
    ]


def test_download_restarts_when_server_returns_wrong_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = tmp_path / "forecast.tar"
    partial_path = target_path.with_suffix(".tar.part")
    partial_path.write_bytes(b"old")
    invalid_range = _response(
        206,
        [b"wrong"],
        {"Content-Range": "bytes 0-4/5"},
    )
    fresh_response = _response(
        200,
        [b"fresh"],
        {"Content-Length": "5"},
    )
    get = MagicMock(side_effect=[invalid_range, fresh_response])
    monkeypatch.setattr(knmi_api.requests, "get", get)
    monkeypatch.setattr(knmi_api.time, "sleep", lambda _: None)

    knmi_api._download_file("https://example.test/file", target_path)

    assert target_path.read_bytes() == b"fresh"
    assert get.call_args_list[1].kwargs["headers"] == {}
    assert not partial_path.exists()


def _interrupted_chunks():
    yield b"abc"
    raise requests.ConnectionError("temporary network loss")
