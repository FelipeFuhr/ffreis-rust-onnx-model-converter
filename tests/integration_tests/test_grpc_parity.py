"""Parity checks between converter HTTP and gRPC contracts/behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, cast, no_type_check

import httpx
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from onnx_converter.converter.core import ConversionOutcome, ConversionRequest
from onnx_converter.converter.http_server import create_app

try:
    import converter_grpc.converter_pb2 as converter_pb2
    import grpc

    from onnx_converter.converter.grpc_server import ConverterGrpcService
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.integration
_HYPOTHESIS_MAX_EXAMPLES = int(os.getenv("HYPOTHESIS_MAX_EXAMPLES", "30"))

_HTTP_TO_GRPC_SURFACE_MAP: dict[str, str] = {
    "/v1/convert/upload": "Convert",
}
_UNMAPPED_HTTP_PATHS: set[str] = {"/healthz", "/readyz"}
_UNMAPPED_GRPC_METHODS: set[str] = set()


@dataclass
class _AbortRecorder:
    """Capture abort arguments from service context."""

    code: str | None = None
    details: str | None = None

    def abort(self, code: object, details: str) -> None:
        """Raise runtime error on abort."""
        self.code = str(getattr(code, "name", code))
        self.details = details
        code_name = getattr(code, "name", "UNKNOWN")
        raise RuntimeError(f"aborted: {code_name}: {details}")


class _DescriptorOwner(Protocol):
    DESCRIPTOR: object


class _FastApiLike(Protocol):
    def openapi(self) -> dict[str, object]: ...


def _fields(message: _DescriptorOwner) -> set[str]:
    """Return protobuf message field names."""
    descriptor_owner = cast(object, message)
    return set(descriptor_owner.DESCRIPTOR.fields_by_name.keys())


def _grpc_method_names() -> set[str]:
    """Return gRPC method names from protobuf service descriptor."""
    service = converter_pb2.DESCRIPTOR.services_by_name["ConverterService"]
    return {method.name for method in service.methods}


def _http_route_paths(application: _FastApiLike) -> set[str]:
    """Return documented HTTP paths from OpenAPI schema."""
    openapi = application.openapi()
    paths = cast(dict[str, object], openapi.get("paths", {}))
    return set(paths.keys())


def _http_upload_form_fields(application: _FastApiLike) -> set[str]:
    """Return multipart form fields for HTTP conversion upload endpoint."""
    openapi = application.openapi()
    paths = cast(dict[str, object], openapi["paths"])
    upload_path = cast(dict[str, object], paths["/v1/convert/upload"])
    post = cast(dict[str, object], upload_path["post"])
    request_body = cast(dict[str, object], post["requestBody"])
    content = cast(dict[str, object], request_body["content"])
    multipart = cast(dict[str, object], content["multipart/form-data"])
    upload_schema = cast(dict[str, str], multipart["schema"])
    schema_ref = upload_schema["$ref"]
    schema_name = schema_ref.split("/")[-1]
    components = cast(dict[str, object], openapi["components"])
    schemas = cast(dict[str, object], components["schemas"])
    schema = cast(dict[str, object], schemas[schema_name])
    properties = cast(dict[str, object], schema["properties"])
    return set(properties.keys())


def test_surface_parity_http_to_grpc_map_is_exhaustive() -> None:
    """Fail when HTTP/gRPC surfaces diverge without explicit mapping."""
    app = create_app()

    discovered_http = _http_route_paths(app)
    discovered_grpc = _grpc_method_names()
    mapped_http = set(_HTTP_TO_GRPC_SURFACE_MAP.keys())
    mapped_grpc = set(_HTTP_TO_GRPC_SURFACE_MAP.values())

    assert discovered_http == mapped_http | _UNMAPPED_HTTP_PATHS
    assert discovered_grpc == mapped_grpc | _UNMAPPED_GRPC_METHODS


def test_schema_mapping_for_convert_request_is_explicit() -> None:
    """Validate HTTP multipart fields map cleanly to gRPC metadata fields."""
    app = create_app()
    http_fields = _http_upload_form_fields(app)
    grpc_fields = _fields(converter_pb2.ConvertMetadata)

    # `artifact` is transport-only in HTTP; gRPC carries raw bytes as chunk data.
    # `filename` comes from uploaded file metadata in HTTP, not an explicit form field.
    # `allow_unsafe` is intentionally not exposed by transport APIs.
    expected_http_fields = (grpc_fields - {"filename"}) | {"artifact"}
    assert http_fields == expected_http_fields


def test_grpc_contract_for_convert_messages() -> None:
    """Validate gRPC request/reply payload contracts."""
    metadata_fields = _fields(converter_pb2.ConvertMetadata)
    request_fields = _fields(converter_pb2.ConvertRequestChunk)
    result_fields = _fields(converter_pb2.ConvertResult)

    assert metadata_fields == {
        "framework",
        "filename",
        "expected_sha256",
        "input_shape",
        "n_features",
        "opset_version",
    }
    assert request_fields == {"metadata", "data"}
    assert result_fields == {
        "input_sha256",
        "output_sha256",
        "output_filename",
        "output_size_bytes",
    }


@pytest.mark.asyncio
async def test_health_and_ready_parity() -> None:
    """Ensure standardized health/readiness endpoints are available."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/healthz")
        ready = await client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_http_and_grpc_success_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare successful conversion outputs between HTTP and gRPC."""
    expected_payload = b"artifact-bytes"

    def fake_convert_artifact_bytes(
        data: bytes,
        request: ConversionRequest,
    ) -> tuple[str, ConversionOutcome]:
        _ = request
        assert data == expected_payload
        return (
            "insha",
            ConversionOutcome(
                output_bytes=b"onnx-binary",
                output_filename="converted.onnx",
                output_sha256="outsha",
                output_size_bytes=len(b"onnx-binary"),
            ),
        )

    import onnx_converter.converter.grpc_server as grpc_module
    import onnx_converter.converter.http_server as http_module

    monkeypatch.setattr(
        http_module,
        "convert_artifact_bytes",
        fake_convert_artifact_bytes,
    )
    monkeypatch.setattr(
        grpc_module,
        "convert_artifact_bytes",
        fake_convert_artifact_bytes,
    )

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/v1/convert/upload",
            files={
                "artifact": (
                    "model.joblib",
                    expected_payload,
                    "application/octet-stream",
                )
            },
            data={"framework": "sklearn", "n_features": "4"},
        )

    service = ConverterGrpcService()
    context = _AbortRecorder()
    grpc_replies = list(
        service.Convert(
            iter(
                [
                    converter_pb2.ConvertRequestChunk(
                        metadata=converter_pb2.ConvertMetadata(
                            framework="sklearn",
                            filename="model.joblib",
                            n_features=4,
                        )
                    ),
                    converter_pb2.ConvertRequestChunk(data=expected_payload),
                ]
            ),
            context,
        )
    )

    assert http_response.status_code == 200
    assert grpc_replies

    grpc_result = grpc_replies[0].result
    grpc_output = b"".join(reply.data for reply in grpc_replies[1:])

    assert http_response.headers["x-input-sha256"] == grpc_result.input_sha256
    assert http_response.headers["x-output-sha256"] == grpc_result.output_sha256
    assert http_response.headers["x-output-filename"] == grpc_result.output_filename
    assert http_response.content == grpc_output


@pytest.mark.asyncio
async def test_http_and_grpc_error_category_parity() -> None:
    """Map missing payload errors to equivalent client-error semantics."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/v1/convert/upload",
            files={"artifact": ("model.bin", b"", "application/octet-stream")},
            data={"framework": "tensorflow"},
        )

    service = ConverterGrpcService()
    context = _AbortRecorder()

    with pytest.raises(RuntimeError, match="missing artifact payload"):
        list(
            service.Convert(
                iter(
                    [
                        converter_pb2.ConvertRequestChunk(
                            metadata=converter_pb2.ConvertMetadata(
                                framework="tensorflow",
                                filename="model.savedmodel",
                            )
                        )
                    ]
                ),
                context,
            )
        )

    assert http_response.status_code == 400
    assert "empty" in http_response.json()["detail"]
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


@pytest.mark.asyncio
async def test_http_and_grpc_error_parity_for_input_sha_mismatch() -> None:
    """Map SHA mismatch errors to equivalent HTTP/gRPC client-error category."""
    payload = b"artifact-bytes"
    wrong_sha = "0" * 64

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/v1/convert/upload",
            files={"artifact": ("model.bin", payload, "application/octet-stream")},
            data={
                "framework": "tensorflow",
                "expected_sha256": wrong_sha,
            },
        )

    service = ConverterGrpcService()
    context = _AbortRecorder()

    with pytest.raises(RuntimeError, match="input SHA-256 mismatch"):
        list(
            service.Convert(
                iter(
                    [
                        converter_pb2.ConvertRequestChunk(
                            metadata=converter_pb2.ConvertMetadata(
                                framework="tensorflow",
                                filename="model.savedmodel",
                                expected_sha256=wrong_sha,
                            )
                        ),
                        converter_pb2.ConvertRequestChunk(data=payload),
                    ]
                ),
                context,
            )
        )

    assert http_response.status_code == 400
    assert "mismatch" in http_response.json()["detail"].lower()
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


@pytest.mark.property
@settings(
    max_examples=_HYPOTHESIS_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    payload=st.binary(min_size=1, max_size=2048),
    framework=st.sampled_from(["sklearn", "tensorflow", "pytorch"]),
    n_features=st.integers(min_value=1, max_value=128),
)
@no_type_check
def test_http_and_grpc_property_parity_for_successful_convert(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    framework: str,
    n_features: int,
) -> None:
    """Property check: transport envelopes stay parity-aligned on success."""

    def fake_convert_artifact_bytes(
        data: bytes,
        request: ConversionRequest,
    ) -> tuple[str, ConversionOutcome]:
        _ = request
        assert data == payload
        return (
            "insha",
            ConversionOutcome(
                output_bytes=b"onnx-binary",
                output_filename="converted.onnx",
                output_sha256="outsha",
                output_size_bytes=len(b"onnx-binary"),
            ),
        )

    import onnx_converter.converter.grpc_server as grpc_module
    import onnx_converter.converter.http_server as http_module

    monkeypatch.setattr(
        http_module, "convert_artifact_bytes", fake_convert_artifact_bytes
    )
    monkeypatch.setattr(
        grpc_module, "convert_artifact_bytes", fake_convert_artifact_bytes
    )

    app = create_app()
    with TestClient(app) as client:
        http_response = client.post(
            "/v1/convert/upload",
            files={"artifact": ("model.bin", payload, "application/octet-stream")},
            data={"framework": framework, "n_features": str(n_features)},
        )

    service = ConverterGrpcService()
    context = _AbortRecorder()
    grpc_replies = list(
        service.Convert(
            iter(
                [
                    converter_pb2.ConvertRequestChunk(
                        metadata=converter_pb2.ConvertMetadata(
                            framework=framework,
                            filename="model.bin",
                            n_features=n_features,
                        )
                    ),
                    converter_pb2.ConvertRequestChunk(data=payload),
                ]
            ),
            context,
        )
    )

    assert http_response.status_code == 200
    assert grpc_replies
    grpc_result = grpc_replies[0].result
    grpc_output = b"".join(reply.data for reply in grpc_replies[1:])

    assert http_response.headers["x-input-sha256"] == grpc_result.input_sha256
    assert http_response.headers["x-output-sha256"] == grpc_result.output_sha256
    assert http_response.headers["x-output-filename"] == grpc_result.output_filename
    assert http_response.content == grpc_output
