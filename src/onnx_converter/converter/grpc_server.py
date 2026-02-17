"""gRPC server for converter daemon artifact transport."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator
from concurrent import futures
from typing import TYPE_CHECKING, Any, cast

import grpc

from converter_grpc import converter_pb2 as _converter_pb2
from converter_grpc import converter_pb2_grpc as _converter_pb2_grpc
from onnx_converter.converter.core import ConversionRequest, convert_artifact_bytes
from onnx_converter.errors import ConversionError

converter_pb2: Any = cast(Any, _converter_pb2)
converter_pb2_grpc: Any = cast(Any, _converter_pb2_grpc)

if TYPE_CHECKING:
    from grpc import ServicerContext
else:
    ServicerContext = Any


def _iter_chunks(payload: bytes, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Yield payload bytes in fixed-size chunks."""
    offset = 0
    total = len(payload)
    while offset < total:
        end = min(offset + chunk_size, total)
        yield payload[offset:end]
        offset = end


class ConverterGrpcService:
    """gRPC converter service implementation."""

    def Convert(
        self,
        request_iterator: Iterable[object],
        context: ServicerContext,
    ) -> Iterator[object]:  # noqa: N802
        """Receive artifact chunks, run conversion, and stream ONNX output chunks."""
        metadata = None
        chunks: list[bytes] = []
        for request in request_iterator:
            chunk = cast(Any, request)
            payload_kind = chunk.WhichOneof("payload")
            if payload_kind == "metadata":
                if metadata is not None:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "metadata provided more than once",
                    )
                metadata = chunk.metadata
                continue
            if payload_kind != "data":
                continue
            data = bytes(chunk.data)
            if data:
                chunks.append(data)

        if metadata is None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "missing conversion metadata",
            )
        if not chunks:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing artifact payload")

        md = cast(Any, metadata)
        framework = str(md.framework).strip().lower()
        input_shape = (
            tuple(int(v) for v in md.input_shape)
            if getattr(md, "input_shape", None)
            else None
        )
        n_features_raw = int(md.n_features) if int(md.n_features) > 0 else None
        request = ConversionRequest(
            framework=framework,  # type: ignore[arg-type]
            filename=str(md.filename or "artifact.bin"),
            expected_sha256=(str(md.expected_sha256).strip() or None),
            input_shape=input_shape,
            n_features=n_features_raw,
            opset_version=int(md.opset_version or 14),
            allow_unsafe=bool(md.allow_unsafe),
        )
        try:
            input_sha, outcome = convert_artifact_bytes(b"".join(chunks), request)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except ConversionError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        yield converter_pb2.ConvertReplyChunk(
            result=converter_pb2.ConvertResult(
                input_sha256=input_sha,
                output_sha256=outcome.output_sha256,
                output_filename=outcome.output_filename,
                output_size_bytes=outcome.output_size_bytes,
            )
        )
        for chunk in _iter_chunks(outcome.output_bytes):
            yield converter_pb2.ConvertReplyChunk(data=chunk)


def create_server(*, host: str, port: int, max_workers: int = 8) -> grpc.Server:
    """Create gRPC server instance."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    converter_pb2_grpc.add_ConverterServiceServicer_to_server(
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
