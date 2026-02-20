"""Unit tests for converter daemon gRPC transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import pytest

from onnx_converter.errors import ConversionError


@dataclass
class _AbortRecorder:
    """Capture abort arguments from service context."""

    code: str | None = None
    details: str | None = None

    def abort(self, code: object, details: str) -> None:
        """Raise runtime error on abort."""
        self.code = str(getattr(code, "name", code))
        self.details = details
        raise RuntimeError(f"aborted: {code.name}: {details}")


class _GrpcModuleLike(Protocol):
    class StatusCode:
        INVALID_ARGUMENT: object


def _deps() -> tuple[_GrpcModuleLike, object, object]:
    grpc = pytest.importorskip("grpc")
    converter_pb2 = pytest.importorskip("converter_grpc.converter_pb2")
    from onnx_converter.converter import grpc_server as module

    return grpc, converter_pb2, module


def test_convert_stream_returns_metadata_and_payload_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream converted output with result metadata as first frame."""
    _, converter_pb2, module = _deps()
    seen: dict[str, object] = {}

    def fake_convert_artifact_bytes(
        data: bytes,
        request: object,
    ) -> tuple[str, object]:
        assert data == b"chunk-a" + b"chunk-b"
        seen["request"] = request
        from onnx_converter.converter.core import ConversionOutcome

        return (
            "insha",
            ConversionOutcome(
                output_bytes=b"onnx-data",
                output_filename="converted.onnx",
                output_sha256="outsha",
                output_size_bytes=len(b"onnx-data"),
            ),
        )

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    service = module.ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter(
        [
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(
                    framework="tensorflow",
                    filename="model.savedmodel",
                    input_shape=[1, 3, 224, 224],
                    n_features=4,
                    opset_version=17,
                )
            ),
            converter_pb2.ConvertRequestChunk(),
            converter_pb2.ConvertRequestChunk(data=b"chunk-a"),
            converter_pb2.ConvertRequestChunk(data=b"chunk-b"),
        ]
    )
    replies = list(service.Convert(request_stream, context))
    assert replies
    first = replies[0]
    assert first.result.input_sha256 == "insha"
    assert first.result.output_sha256 == "outsha"
    assert first.result.output_filename == "converted.onnx"
    data = b"".join(chunk.data for chunk in replies[1:])
    assert data == b"onnx-data"
    from onnx_converter.converter.core import ConversionRequest

    request = cast(ConversionRequest, seen["request"])
    assert request.framework == "tensorflow"
    assert request.input_shape == (1, 3, 224, 224)
    assert request.n_features == 4
    assert request.opset_version == 17
    assert request.allow_unsafe is False


def test_convert_stream_rejects_missing_metadata() -> None:
    """Abort request when stream does not include metadata frame."""
    grpc, converter_pb2, module = _deps()
    service = module.ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter([converter_pb2.ConvertRequestChunk(data=b"x")])
    with pytest.raises(RuntimeError, match="missing conversion metadata"):
        list(service.Convert(request_stream, context))
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


def test_convert_stream_rejects_duplicate_metadata() -> None:
    """Abort request when metadata frame appears more than once."""
    grpc, converter_pb2, module = _deps()
    service = module.ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter(
        [
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(framework="tensorflow")
            ),
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(framework="tensorflow")
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="metadata provided more than once"):
        list(service.Convert(request_stream, context))
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


def test_convert_stream_rejects_missing_payload() -> None:
    """Abort request when no data chunks are provided."""
    grpc, converter_pb2, module = _deps()
    service = module.ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter(
        [
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(framework="tensorflow")
            )
        ]
    )
    with pytest.raises(RuntimeError, match="missing artifact payload"):
        list(service.Convert(request_stream, context))
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


def test_convert_stream_maps_value_error_to_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map ValueError from conversion to INVALID_ARGUMENT."""
    grpc, converter_pb2, module = _deps()

    def fake_convert_artifact_bytes(
        data: bytes,
        request: object,
    ) -> tuple[str, object]:
        _ = (data, request)
        raise ValueError("invalid request")

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    service = module.ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter(
        [
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(framework="tensorflow")
            ),
            converter_pb2.ConvertRequestChunk(data=b"x"),
        ]
    )
    with pytest.raises(RuntimeError, match="invalid request"):
        list(service.Convert(request_stream, context))
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


def test_convert_stream_maps_conversion_error_to_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map conversion domain errors to INVALID_ARGUMENT."""
    grpc, converter_pb2, module = _deps()

    def fake_convert_artifact_bytes(
        data: bytes,
        request: object,
    ) -> tuple[str, object]:
        _ = (data, request)
        raise ConversionError("conversion exploded")

    monkeypatch.setattr(module, "convert_artifact_bytes", fake_convert_artifact_bytes)
    service = module.ConverterGrpcService()
    context = _AbortRecorder()
    request_stream = iter(
        [
            converter_pb2.ConvertRequestChunk(
                metadata=converter_pb2.ConvertMetadata(framework="tensorflow")
            ),
            converter_pb2.ConvertRequestChunk(data=b"x"),
        ]
    )
    with pytest.raises(RuntimeError, match="conversion exploded"):
        list(service.Convert(request_stream, context))
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT.name


def test_iter_chunks_splits_payload() -> None:
    """Split payload using provided chunk size."""
    _, _, module = _deps()
    chunks = list(module._iter_chunks(b"abcdef", chunk_size=2))
    assert chunks == [b"ab", b"cd", b"ef"]


def test_create_server_registers_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create server and bind host/port."""
    _, _, module = _deps()
    calls: dict[str, object] = {}

    class _FakeServer:
        def add_insecure_port(self, address: str) -> None:
            calls["address"] = address

    fake_server = _FakeServer()

    def fake_grpc_server(executor: object) -> _FakeServer:
        calls["executor"] = executor
        return fake_server

    def fake_register(servicer: object, server: object) -> None:
        calls["servicer"] = servicer
        calls["server"] = server

    monkeypatch.setattr(module.grpc, "server", fake_grpc_server)
    monkeypatch.setattr(
        module.converter_pb2_grpc,
        "add_ConverterServiceServicer_to_server",
        fake_register,
    )
    created = module.create_server(host="127.0.0.1", port=9000, max_workers=2)
    assert created is fake_server
    assert calls["address"] == "127.0.0.1:9000"
    assert calls["server"] is fake_server
    assert isinstance(calls["servicer"], module.ConverterGrpcService)


def test_main_starts_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Parse args, start server, and wait for termination."""
    _, _, module = _deps()
    started = {"start": False, "wait": False}

    class _Server:
        def start(self) -> None:
            started["start"] = True

        def wait_for_termination(self) -> None:
            started["wait"] = True

    monkeypatch.setattr(
        module,
        "create_server",
        lambda **kwargs: _Server(),
    )
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",  # noqa: ARG005
        lambda self: module.argparse.Namespace(
            host="0.0.0.0",
            port=8091,
            max_workers=8,
        ),
    )
    module.main()
    out = capsys.readouterr().out
    assert "converter grpc listening on 0.0.0.0:8091" in out
    assert started["start"] is True
    assert started["wait"] is True
