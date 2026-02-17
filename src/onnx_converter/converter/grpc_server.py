"""gRPC server for converter daemon artifact transport."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from collections.abc import Iterable, Iterator
from concurrent import futures
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

try:
    import grpc
except ModuleNotFoundError:  # pragma: no cover
    grpc = None

from onnx_converter.converter.core import ConversionRequest, convert_artifact_bytes
from onnx_converter.errors import ConversionError

logger = logging.getLogger(__name__)


def _load_converter_pb2() -> ModuleType | None:
    """Load generated protobuf messages module when present."""
    try:
        return importlib.import_module("converter_grpc.converter_pb2")
    except ModuleNotFoundError:
        return None


def _load_converter_pb2_grpc() -> ModuleType | None:
    """Load generated grpc service stubs module when present."""
    try:
        return importlib.import_module("converter_grpc.converter_pb2_grpc")
    except ModuleNotFoundError:
        return None


converter_pb2 = _load_converter_pb2()
converter_pb2_grpc = _load_converter_pb2_grpc()

if TYPE_CHECKING:
    from grpc import ServicerContext
else:

    class ServicerContext(Protocol):
        """Subset of grpc context API used by this module."""

        def abort(self, code: object, details: str) -> None:
            """Abort RPC with code/details."""
            ...


class _GrpcStatusCode(Protocol):
    """Subset of grpc.StatusCode enum used by this module."""

    INVALID_ARGUMENT: object
    INTERNAL: object


class _GrpcServer(Protocol):
    """Subset of grpc.Server API used by this module."""

    def add_insecure_port(self, address: str) -> int:
        """Bind server to an insecure address."""
        ...

    def start(self) -> None:
        """Start serving requests."""
        ...

    def wait_for_termination(self) -> None:
        """Block until server termination."""
        ...


class _GrpcRuntime(Protocol):
    """Subset of grpc module API used by this module."""

    StatusCode: _GrpcStatusCode

    def server(self, executor: futures.Executor) -> _GrpcServer:
        """Build grpc server instance."""
        ...


class _GrpcStubs(Protocol):
    """Subset of generated stub helpers used by this module."""

    def add_ConverterServiceServicer_to_server(
        self,
        servicer: object,
        server: _GrpcServer,
    ) -> None:
        """Register converter service implementation."""
        ...


class _ConvertMetadataLike(Protocol):
    """Subset of ConvertMetadata fields used by transport handler."""

    framework: str
    filename: str
    expected_sha256: str
    input_shape: Iterable[int]
    n_features: int
    opset_version: int


class _ConvertRequestChunkLike(Protocol):
    """Subset of ConvertRequestChunk API used by transport handler."""

    metadata: _ConvertMetadataLike
    data: bytes

    def WhichOneof(self, group: str) -> str | None:
        """Return active oneof field name."""
        ...


class _ConvertResultLike(Protocol):
    """Marker protocol for generated ConvertResult messages."""


class _ConvertReplyChunkLike(Protocol):
    """Marker protocol for generated ConvertReplyChunk messages."""


class _ConvertResultFactory(Protocol):
    """Constructor signature for generated ConvertResult."""

    def __call__(
        self,
        *,
        input_sha256: str,
        output_sha256: str,
        output_filename: str,
        output_size_bytes: int,
    ) -> _ConvertResultLike:
        """Build ConvertResult."""
        ...


class _ConvertReplyChunkFactory(Protocol):
    """Constructor signature for generated ConvertReplyChunk."""

    def __call__(
        self,
        *,
        result: _ConvertResultLike | None = None,
        data: bytes = b"",
    ) -> _ConvertReplyChunkLike:
        """Build ConvertReplyChunk."""
        ...


class _ConverterPb2(Protocol):
    """Subset of generated pb2 module API used by this module."""

    ConvertResult: _ConvertResultFactory
    ConvertReplyChunk: _ConvertReplyChunkFactory


def _require_grpc_runtime() -> _GrpcRuntime:
    """Return grpc module or raise an actionable runtime error."""
    if grpc is None:
        raise RuntimeError(
            "grpcio is required to run converter-grpc. Install with extra: .[grpc]"
        )
    return cast(_GrpcRuntime, grpc)


def _require_grpc_stubs() -> _GrpcStubs:
    """Return generated grpc stubs module or raise an actionable runtime error."""
    if converter_pb2_grpc is None:
        raise RuntimeError(
            "gRPC stubs are unavailable. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return cast(_GrpcStubs, converter_pb2_grpc)


def _require_converter_pb2() -> _ConverterPb2:
    """Return generated protobuf messages module or raise actionable error."""
    if converter_pb2 is None:
        raise RuntimeError(
            "gRPC protobuf messages are unavailable. "
            "Run ./scripts/generate_grpc_stubs.sh first."
        )
    return cast(_ConverterPb2, converter_pb2)


def _iter_chunks(payload: bytes, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Yield payload bytes in fixed-size chunks."""
    offset = 0
    total = len(payload)
    while offset < total:
        end = min(offset + chunk_size, total)
        yield payload[offset:end]
        offset = end


def _collect_convert_payload(
    request_iterator: Iterable[_ConvertRequestChunkLike],
    context: ServicerContext,
    grpc_runtime: _GrpcRuntime,
) -> tuple[_ConvertMetadataLike, list[bytes]]:
    """Collect metadata and non-empty data chunks from request stream."""
    metadata: _ConvertMetadataLike | None = None
    chunks: list[bytes] = []
    for request_chunk in request_iterator:
        payload_kind = request_chunk.WhichOneof("payload")
        if payload_kind == "metadata":
            if metadata is not None:
                context.abort(
                    grpc_runtime.StatusCode.INVALID_ARGUMENT,
                    "metadata provided more than once",
                )
            metadata = request_chunk.metadata
            continue
        if payload_kind != "data":
            continue
        data = bytes(request_chunk.data)
        if data:
            chunks.append(data)

    if metadata is None:
        context.abort(
            grpc_runtime.StatusCode.INVALID_ARGUMENT,
            "missing conversion metadata",
        )
    if not chunks:
        context.abort(
            grpc_runtime.StatusCode.INVALID_ARGUMENT,
            "missing artifact payload",
        )
    assert metadata is not None
    return metadata, chunks


def _build_conversion_request(metadata: _ConvertMetadataLike) -> ConversionRequest:
    """Build domain conversion request from transport metadata."""
    framework = str(metadata.framework).strip().lower()
    input_shape = (
        tuple(int(v) for v in metadata.input_shape) if metadata.input_shape else None
    )
    n_features_raw = int(metadata.n_features) if int(metadata.n_features) > 0 else None
    return ConversionRequest(
        framework=framework,  # type: ignore[arg-type]
        filename=str(metadata.filename or "artifact.bin"),
        expected_sha256=(str(metadata.expected_sha256).strip() or None),
        input_shape=input_shape,
        n_features=n_features_raw,
        opset_version=int(metadata.opset_version or 14),
        # gRPC transport does not allow clients to opt into unsafe deserialization.
        allow_unsafe=False,
    )


def _stream_convert_reply(
    pb2: _ConverterPb2,
    *,
    input_sha: str,
    output_sha256: str,
    output_filename: str,
    output_size_bytes: int,
    output_bytes: bytes,
) -> Iterator[_ConvertReplyChunkLike]:
    """Yield result metadata frame followed by payload chunks."""
    yield pb2.ConvertReplyChunk(
        result=pb2.ConvertResult(
            input_sha256=input_sha,
            output_sha256=output_sha256,
            output_filename=output_filename,
            output_size_bytes=output_size_bytes,
        )
    )
    for chunk in _iter_chunks(output_bytes):
        yield pb2.ConvertReplyChunk(data=chunk)


class ConverterGrpcService:
    """gRPC converter service implementation."""

    def convert(
        self,
        request_iterator: Iterable[_ConvertRequestChunkLike],
        context: ServicerContext,
    ) -> Iterator[_ConvertReplyChunkLike]:
        """Receive artifact chunks, run conversion, and stream ONNX output chunks."""
        grpc_runtime = _require_grpc_runtime()
        pb2 = _require_converter_pb2()
        metadata, chunks = _collect_convert_payload(
            request_iterator,
            context,
            grpc_runtime,
        )
        conversion_request = _build_conversion_request(metadata)
        try:
            input_sha, outcome = convert_artifact_bytes(
                b"".join(chunks),
                conversion_request,
            )
        except ValueError as exc:
            context.abort(grpc_runtime.StatusCode.INVALID_ARGUMENT, str(exc))
        except ConversionError as exc:
            context.abort(grpc_runtime.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            logger.exception("unexpected error during gRPC conversion")
            context.abort(grpc_runtime.StatusCode.INTERNAL, "internal server error")
        yield from _stream_convert_reply(
            pb2,
            input_sha=input_sha,
            output_sha256=outcome.output_sha256,
            output_filename=outcome.output_filename,
            output_size_bytes=outcome.output_size_bytes,
            output_bytes=outcome.output_bytes,
        )

    # gRPC generated handlers call the RPC method name "Convert".
    Convert = convert


def create_server(*, host: str, port: int, max_workers: int = 8) -> _GrpcServer:
    """Create gRPC server instance."""
    grpc_runtime = _require_grpc_runtime()
    stubs = _require_grpc_stubs()
    server = grpc_runtime.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    stubs.add_ConverterServiceServicer_to_server(
        ConverterGrpcService(),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server


def main() -> None:
    """Run converter daemon gRPC entrypoint."""
    parser = argparse.ArgumentParser(description="ONNX converter daemon gRPC server.")
    parser.add_argument(
        "--host",
        default=os.getenv("CONVERTER_GRPC_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CONVERTER_GRPC_PORT", "8091")),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    server = create_server(host=args.host, port=args.port, max_workers=args.max_workers)
    server.start()
    print(f"converter grpc listening on {args.host}:{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
