"""HTTP server for artifact upload/download conversion."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel, ConfigDict

from onnx_converter.converter.core import (
    ConversionRequest,
    Framework,
    convert_artifact_bytes,
)
from onnx_converter.errors import ConversionError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import Response
else:

    class UploadFile:
        """Fallback UploadFile type used when FastAPI is not installed."""

        filename: str | None = None

        async def read(self) -> bytes:
            """Read uploaded content bytes."""
            return b""


class _FastapiStatusLike(Protocol):
    HTTP_400_BAD_REQUEST: int
    HTTP_500_INTERNAL_SERVER_ERROR: int


class _FastapiModuleLike(Protocol):
    """Subset of fastapi module API used by HTTP transport."""

    status: _FastapiStatusLike

    class HTTPException(Exception):
        def __init__(self, *, status_code: int, detail: str) -> None: ...

    def File(self, default: object) -> object: ...

    def Form(self, default: object = ..., **kwargs: object) -> object: ...

    def FastAPI(self, **kwargs: object) -> object: ...


class _ResponsesModuleLike(Protocol):
    """Subset of fastapi.responses module API used by this module."""

    def Response(
        self,
        *,
        content: bytes,
        media_type: str,
        headers: dict[str, str],
    ) -> object: ...


_fastapi_module: ModuleType | None = None
_fastapi_responses_module: ModuleType | None = None
try:
    _fastapi_module = importlib.import_module("fastapi")
    _fastapi_responses_module = importlib.import_module("fastapi.responses")
except ModuleNotFoundError:  # pragma: no cover
    pass

if _fastapi_module is not None and not TYPE_CHECKING:
    UploadFile = cast(type[UploadFile], _fastapi_module.UploadFile)

fastapi = cast(_FastapiModuleLike | None, _fastapi_module)
responses = _fastapi_responses_module

try:
    import uvicorn as _uvicorn_imported

    _uvicorn_module: ModuleType | None = _uvicorn_imported
except ModuleNotFoundError:  # pragma: no cover
    _uvicorn_module = None

uvicorn: ModuleType | None = _uvicorn_module


def _require_http_runtime() -> None:
    """Ensure HTTP server runtime dependencies are available."""
    if _fastapi_module is None or _fastapi_responses_module is None:
        raise RuntimeError(
            "fastapi is required to run converter-http. Install with extra: .[server]"
        )


class HealthResponse(BaseModel):
    """Health response payload."""

    model_config = ConfigDict(extra="forbid")

    status: str


class ReadyResponse(BaseModel):
    """Readiness response payload."""

    model_config = ConfigDict(extra="forbid")

    status: str


def _parse_input_shape(value: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated input shape string."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        dims = tuple(int(part.strip()) for part in cleaned.split(","))
    except ValueError as exc:
        raise ValueError("input_shape must be comma-separated integers") from exc
    if not dims:
        return None
    return dims


def _parse_framework(value: str) -> Framework:
    """Normalize and validate framework for transport payloads."""
    normalized = value.strip().lower()
    if normalized not in {"pytorch", "tensorflow", "sklearn"}:
        raise ValueError("framework must be one of: pytorch, tensorflow, sklearn")
    return cast(Framework, normalized)


def create_app() -> FastAPI:
    """Create converter daemon HTTP application."""
    _require_http_runtime()
    if fastapi is None:
        raise RuntimeError("fastapi module is unavailable")
    fastapi_module = fastapi
    responses_module = cast(_ResponsesModuleLike, responses)
    app = cast(
        "FastAPI",
        fastapi_module.FastAPI(
            title="ONNX Converter Daemon",
            version="0.1.0",
            description=(
                "Upload model artifacts and download converted ONNX with integrity "
                "checks."
            ),
        ),
    )
    artifact_param = cast(UploadFile, fastapi_module.File(...))
    framework_param = cast(str, fastapi_module.Form(...))
    expected_sha256_param = cast(str | None, fastapi_module.Form(default=None))
    input_shape_param = cast(str | None, fastapi_module.Form(default=None))
    n_features_param = cast(int | None, fastapi_module.Form(default=None))
    opset_version_param = cast(int, fastapi_module.Form(default=14))

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz() -> ReadyResponse:
        return ReadyResponse(status="ready")

    @app.post("/v1/convert/upload")
    async def convert_upload(
        artifact: UploadFile = artifact_param,
        framework: str = framework_param,
        expected_sha256: str | None = expected_sha256_param,
        input_shape: str | None = input_shape_param,
        n_features: int | None = n_features_param,
        opset_version: int = opset_version_param,
    ) -> Response:
        """Convert uploaded artifact and return ONNX bytes."""
        artifact_name = artifact.filename or "artifact.bin"
        payload = await artifact.read()
        if not payload:
            raise fastapi_module.HTTPException(
                status_code=fastapi_module.status.HTTP_400_BAD_REQUEST,
                detail="uploaded artifact is empty",
            )
        try:
            normalized_framework = framework.strip().lower()
            request = ConversionRequest(
                framework=_parse_framework(normalized_framework),
                filename=artifact_name,
                expected_sha256=expected_sha256,
                input_shape=_parse_input_shape(input_shape),
                n_features=n_features,
                opset_version=opset_version,
                # Unsafe deserialization is intentionally disabled in transport APIs.
                allow_unsafe=False,
            )
            input_sha, outcome = convert_artifact_bytes(payload, request)
        except (ValueError, ConversionError) as exc:
            raise fastapi_module.HTTPException(
                status_code=fastapi_module.status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover
            logger.exception("unexpected error during HTTP conversion upload")
            raise fastapi_module.HTTPException(
                status_code=fastapi_module.status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="internal server error",
            ) from exc

        headers = {
            "X-Input-SHA256": input_sha,
            "X-Output-SHA256": outcome.output_sha256,
            "X-Output-Filename": outcome.output_filename,
            "Content-Disposition": f'attachment; filename="{outcome.output_filename}"',
        }
        return cast(
            "Response",
            responses_module.Response(
                content=outcome.output_bytes,
                media_type="application/octet-stream",
                headers=headers,
            ),
        )

    return app


if TYPE_CHECKING:
    app: FastAPI | None

if _fastapi_module is not None:
    app = create_app()
else:  # pragma: no cover
    app = None


def main() -> None:
    """Run converter daemon HTTP entrypoint."""
    _require_http_runtime()
    if uvicorn is None:
        raise RuntimeError("uvicorn is required to run converter-http")
    parser = argparse.ArgumentParser(description="ONNX converter daemon HTTP server.")
    parser.add_argument(
        "--host",
        default=os.getenv("CONVERTER_HTTP_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CONVERTER_HTTP_PORT", "8090")),
    )
    args = parser.parse_args()
    uvicorn.run(
        "onnx_converter.converter.http_server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
