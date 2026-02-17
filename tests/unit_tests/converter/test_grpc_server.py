"""Unit tests for converter daemon gRPC transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import grpc
import pytest

from converter_grpc import converter_pb2
from onnx_converter.converter.core import ConversionOutcome
from onnx_converter.converter.grpc_server import ConverterGrpcService


@dataclass
class _AbortRecorder:
    """Capture abort arguments from service context."""

    code: grpc.StatusCode | None = None
    details: str | None = None

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        """Raise runtime error on abort."""
        self.code = code
        self.details = details
        raise RuntimeError(f"aborted: {code.name}: {details}")


def test_convert_stream_returns_metadata_and_payload_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream converted output with result metadata as first frame."""

    def fake_convert_artifact_bytes(
        data: bytes,
        request: object,
    ) -> tuple[str, ConversionOutcome]:
        _ = request
        assert data == b"chunk-a" + b"chunk-b"
        return (
            "insha",
            ConversionOutcome(
                output_bytes=b"onnx-data",
                output_filename="converted.onnx",
                output_sha256="outsha",
                output_size_bytes=len(b"onnx-data"),
            ),
        )

    import onnx_converter.converter.grpc_server as module

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    service = ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter(
        [
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(
                    framework="tensorflow",
                    filename="model.savedmodel",
                )
            ),
            converter_pb2.ConvertRequestChunk(data=b"chunk-a"),
            converter_pb2.ConvertRequestChunk(data=b"chunk-b"),
        ]
    )
    replies = list(service.Convert(request_stream, cast(Any, context)))
    assert replies
    first = replies[0]
    assert first.result.input_sha256 == "insha"
    assert first.result.output_sha256 == "outsha"
    assert first.result.output_filename == "converted.onnx"
    data = b"".join(chunk.data for chunk in replies[1:])
    assert data == b"onnx-data"


def test_convert_stream_rejects_missing_metadata() -> None:
    """Abort request when stream does not include metadata frame."""
    service = ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter([converter_pb2.ConvertRequestChunk(data=b"x")])
    with pytest.raises(RuntimeError, match="missing conversion metadata"):
        list(service.Convert(request_stream, cast(Any, context)))
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
