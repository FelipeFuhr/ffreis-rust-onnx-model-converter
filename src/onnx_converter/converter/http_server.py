"""HTTP server for artifact upload/download conversion."""

from __future__ import annotations

import argparse
import os
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from onnx_converter.converter.core import ConversionRequest, convert_artifact_bytes
from onnx_converter.errors import ConversionError


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


def create_app() -> FastAPI:
    """Create converter daemon HTTP application."""
    app = FastAPI(
        title="ONNX Converter Daemon",
        version="0.1.0",
        description=(
            "Upload model artifacts and download converted ONNX with integrity checks."
        ),
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz() -> ReadyResponse:
        return ReadyResponse(status="ready")

    @app.post("/v1/convert/upload")
    async def convert_upload(
        artifact: Annotated[UploadFile, File(...)],
        framework: Annotated[str, Form(...)],
        expected_sha256: Annotated[str | None, Form()] = None,
        input_shape: Annotated[str | None, Form()] = None,
        n_features: Annotated[int | None, Form()] = None,
        opset_version: Annotated[int, Form()] = 14,
        allow_unsafe: Annotated[bool, Form()] = False,
    ) -> Response:
        """Convert uploaded artifact and return ONNX bytes."""
        artifact_name = artifact.filename or "artifact.bin"
        payload = await artifact.read()
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="uploaded artifact is empty",
            )
        try:
            request = ConversionRequest(
                framework=framework,  # type: ignore[arg-type]
                filename=artifact_name,
                expected_sha256=expected_sha256,
                input_shape=_parse_input_shape(input_shape),
                n_features=n_features,
                opset_version=opset_version,
                allow_unsafe=allow_unsafe,
            )
            input_sha, outcome = convert_artifact_bytes(payload, request)
        except (ValueError, ConversionError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        headers = {
            "X-Input-SHA256": input_sha,
            "X-Output-SHA256": outcome.output_sha256,
            "X-Output-Filename": outcome.output_filename,
            "Content-Disposition": f'attachment; filename="{outcome.output_filename}"',
        }
        return Response(
            content=outcome.output_bytes,
            media_type="application/octet-stream",
            headers=headers,
        )

    return app


app = create_app()


def main() -> None:
    """Run converter daemon HTTP entrypoint."""
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
