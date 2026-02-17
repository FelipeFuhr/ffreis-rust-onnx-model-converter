"""Unit tests for converter daemon HTTP transport."""

from __future__ import annotations

import httpx
import pytest

from onnx_converter.converter.core import ConversionOutcome
from onnx_converter.converter.http_server import create_app


@pytest.mark.asyncio
async def test_convert_upload_returns_binary_and_integrity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return converted binary plus input/output digest metadata."""

    def fake_convert_artifact_bytes(
        data: bytes,
        request: object,
    ) -> tuple[str, ConversionOutcome]:
        _ = request
        assert data == b"artifact-bytes"
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
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/convert/upload",
            files={
                "artifact": (
                    "model.joblib",
                    b"artifact-bytes",
                    "application/octet-stream",
                )
            },
            data={"framework": "sklearn", "n_features": "4"},
        )

    assert response.status_code == 200
    assert response.content == b"onnx-binary"
    assert response.headers["x-input-sha256"] == "insha"
    assert response.headers["x-output-sha256"] == "outsha"
    assert response.headers["x-output-filename"] == "converted.onnx"


@pytest.mark.asyncio
async def test_convert_upload_rejects_empty_payload() -> None:
    """Reject zero-byte uploads with a client error."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/convert/upload",
            files={"artifact": ("model.bin", b"", "application/octet-stream")},
            data={"framework": "tensorflow"},
        )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_health_and_ready_endpoints() -> None:
    """Expose liveness and readiness endpoints with 200 responses."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/healthz")
        ready = await client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
