"""Unit tests for converter daemon HTTP transport."""

from __future__ import annotations

import argparse
import types
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from onnx_converter.converter.core import ConversionOutcome, ConversionRequest
from onnx_converter.errors import ConversionError

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class _RequestWithFramework(Protocol):
    framework: str


class _UvicornLike(Protocol):
    def run(self, app_ref: str, *, host: str, port: int, reload: bool) -> None: ...


def _client() -> TestClient:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi.testclient import TestClient

    module = __import__("onnx_converter.converter.http_server", fromlist=["create_app"])

    return TestClient(module.create_app())


def test_convert_upload_returns_binary_and_integrity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return converted binary plus input/output digest metadata."""
    seen: dict[str, _RequestWithFramework] = {}

    def fake_convert_artifact_bytes(
        data: bytes,
        request: ConversionRequest,
    ) -> tuple[str, ConversionOutcome]:
        assert data == b"artifact-bytes"
        seen["request"] = cast(_RequestWithFramework, request)

        return (
            "insha",
            ConversionOutcome(
                output_bytes=b"onnx-binary",
                output_filename="converted.onnx",
                output_sha256="outsha",
                output_size_bytes=len(b"onnx-binary"),
            ),
        )

    import onnx_converter.converter.http_server as module

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    client = _client()
    response = client.post(
        "/v1/convert/upload",
        files={
            "artifact": (
                "model.joblib",
                b"artifact-bytes",
                "application/octet-stream",
            )
        },
        data={"framework": "sklearn", "input_shape": "1,2", "n_features": "4"},
    )

    assert response.status_code == 200
    assert response.content == b"onnx-binary"
    assert response.headers["x-input-sha256"] == "insha"
    assert response.headers["x-output-sha256"] == "outsha"
    assert response.headers["x-output-filename"] == "converted.onnx"
    assert (
        'attachment; filename="converted.onnx"'
        in response.headers["content-disposition"]
    )
    request = seen["request"]
    assert request.framework == "sklearn"


def test_convert_upload_normalizes_framework_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize framework similarly to gRPC transport."""
    seen: dict[str, str] = {}

    def fake_convert_artifact_bytes(
        data: bytes,
        request: ConversionRequest,
    ) -> tuple[str, ConversionOutcome]:
        _ = data
        seen["framework"] = request.framework

        return (
            "insha",
            ConversionOutcome(
                output_bytes=b"onnx-binary",
                output_filename="converted.onnx",
                output_sha256="outsha",
                output_size_bytes=len(b"onnx-binary"),
            ),
        )

    import onnx_converter.converter.http_server as module

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    client = _client()
    response = client.post(
        "/v1/convert/upload",
        files={
            "artifact": ("model.bin", b"artifact-bytes", "application/octet-stream")
        },
        data={"framework": "  SKLEARN  ", "n_features": "4"},
    )
    assert response.status_code == 200
    assert seen["framework"] == "sklearn"


def test_convert_upload_rejects_empty_payload() -> None:
    """Reject zero-byte uploads with a client error."""
    client = _client()
    response = client.post(
        "/v1/convert/upload",
        files={"artifact": ("model.bin", b"", "application/octet-stream")},
        data={"framework": "tensorflow"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_convert_upload_maps_domain_validation_errors_to_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a 400 when conversion layer raises expected validation errors."""

    def fake_convert_artifact_bytes(
        data: bytes,
        request: ConversionRequest,
    ) -> tuple[str, ConversionOutcome]:
        _ = (data, request)
        raise ConversionError("bad artifact")

    import onnx_converter.converter.http_server as module

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    client = _client()
    response = client.post(
        "/v1/convert/upload",
        files={"artifact": ("model.bin", b"x", "application/octet-stream")},
        data={"framework": "tensorflow"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "bad artifact"


def test_convert_upload_maps_unexpected_errors_to_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a 500 when conversion layer raises an unexpected error."""

    def fake_convert_artifact_bytes(
        data: bytes,
        request: ConversionRequest,
    ) -> tuple[str, ConversionOutcome]:
        _ = (data, request)
        raise RuntimeError("unexpected failure")

    import onnx_converter.converter.http_server as module

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    client = _client()
    response = client.post(
        "/v1/convert/upload",
        files={"artifact": ("model.bin", b"x", "application/octet-stream")},
        data={"framework": "tensorflow"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "internal server error"


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("1,2,3", (1, 2, 3)),
    ],
)
def test_parse_input_shape(raw: str | None, parsed: tuple[int, ...] | None) -> None:
    """Parse optional comma-separated input shape."""
    from onnx_converter.converter.http_server import _parse_input_shape

    assert _parse_input_shape(raw) == parsed


def test_parse_input_shape_rejects_invalid_value() -> None:
    """Reject malformed input_shape values."""
    from onnx_converter.converter.http_server import _parse_input_shape

    with pytest.raises(ValueError, match="comma-separated integers"):
        _parse_input_shape("1,a,3")


def test_health_and_ready_endpoints() -> None:
    """Expose liveness and readiness endpoints with 200 responses."""
    client = _client()
    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_main_runs_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse CLI args and pass them to uvicorn."""
    pytest.importorskip("fastapi")
    calls: dict[str, str | int | bool] = {}

    import onnx_converter.converter.http_server as module

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",  # noqa: ARG005
        lambda self: argparse.Namespace(host="127.0.0.1", port=9999),
    )

    def fake_run(app_ref: str, *, host: str, port: int, reload: bool) -> None:
        calls["app_ref"] = app_ref
        calls["host"] = host
        calls["port"] = port
        calls["reload"] = reload

    monkeypatch.setattr(
        module,
        "uvicorn",
        cast(_UvicornLike, types.SimpleNamespace(run=fake_run)),
    )
    module.main()
    assert calls == {
        "app_ref": "onnx_converter.converter.http_server:app",
        "host": "127.0.0.1",
        "port": 9999,
        "reload": False,
    }
