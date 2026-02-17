from __future__ import annotations

import json
import os
import time
import urllib.request

import grpc

from converter_grpc import converter_pb2


def _wait_http_ok(url: str, timeout_seconds: float = 40.0) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:  # noqa: S310
                if response.status == 200:
                    return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for HTTP 200 at {url}: {last_error}")


def _assert_http(api_base: str) -> None:
    health_raw = _wait_http_ok(f"{api_base}/healthz")
    ready_raw = _wait_http_ok(f"{api_base}/readyz")
    health_payload = json.loads(health_raw.decode("utf-8"))
    ready_payload = json.loads(ready_raw.decode("utf-8"))
    assert health_payload.get("status") == "ok", health_payload
    assert ready_payload.get("status") == "ready", ready_payload


def _assert_grpc(grpc_target: str) -> None:
    with grpc.insecure_channel(grpc_target) as channel:
        convert_rpc = channel.stream_stream(
            "/converter.grpc.ConverterService/Convert",
            request_serializer=converter_pb2.ConvertRequestChunk.SerializeToString,
            response_deserializer=converter_pb2.ConvertReplyChunk.FromString,
        )
        requests = iter(
            [
                converter_pb2.ConvertRequestChunk(
                    metadata=converter_pb2.ConvertMetadata(
                        framework="sklearn",
                        filename="dummy.pkl",
                        n_features=4,
                    )
                ),
            ]
        )
        try:
            list(convert_rpc(requests, timeout=5.0))
        except grpc.RpcError as exc:
            assert exc.code() == grpc.StatusCode.INVALID_ARGUMENT, exc
            assert "missing artifact payload" in (exc.details() or ""), exc
            return
    raise AssertionError("expected INVALID_ARGUMENT for missing artifact payload")


def main() -> None:
    api_base = os.getenv("CONVERTER_API_BASE", "http://converter-api:8090")
    grpc_target = os.getenv("CONVERTER_GRPC_TARGET", "converter-grpc:8091")

    _assert_http(api_base)
    _assert_grpc(grpc_target)

    print("converter API and gRPC smoke checks passed")


if __name__ == "__main__":
    main()
