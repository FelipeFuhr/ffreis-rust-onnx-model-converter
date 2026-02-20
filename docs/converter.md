# Converter Transport API

`converter` accepts model artifacts over HTTP or gRPC, verifies optional
input SHA-256 integrity, converts to ONNX, and returns ONNX bytes plus output
integrity metadata.

## Install extras

```bash
# HTTP only
uv sync --extra server

# gRPC only
uv sync --extra grpc

# both interfaces
uv sync --extra server --extra grpc
```

## HTTP mode

Run:

```bash
converter-http --host 0.0.0.0 --port 8090
```

Endpoint:

- `GET /healthz`
- `GET /readyz`
- `POST /v1/convert/upload` (`multipart/form-data`)
  - file field: `artifact`
  - fields:
    - `framework`: `pytorch|tensorflow|sklearn`
    - `expected_sha256` (optional)
    - `input_shape` (optional, comma-separated; required for `pytorch`)
    - `n_features` (optional; required for `sklearn`)
    - `opset_version` (optional, default `14`)
  - notes:
    - `framework` is normalized via `strip().lower()`
    - unsafe filenames are sanitized to a safe basename before temp-file writes
    - unsafe deserialization toggles are not exposed in HTTP/gRPC transport APIs

Response:

- body: ONNX bytes (`application/octet-stream`)
- headers:
  - `X-Input-SHA256`
  - `X-Output-SHA256`
  - `X-Output-Filename`

HTTP error mapping:

- `400` for request/validation/conversion domain errors (`ValueError`, `ConversionError`)
- `500` for unexpected server errors (generic detail: `internal server error`)

## gRPC mode

Run:

```bash
converter-grpc --host 0.0.0.0 --port 8091
```

Proto:

- `proto/converter_grpc/converter.proto`

Flow:

1. client sends one metadata frame (`ConvertMetadata`)
2. client streams artifact data frames (`bytes data`)
3. server validates digest (if provided), converts, sends:
   - first reply frame with `ConvertResult` metadata
   - following reply frames as ONNX byte chunks

gRPC notes:

- `framework` is normalized via `strip().lower()` before conversion
- request/validation/conversion domain errors map to `INVALID_ARGUMENT`
- unexpected server errors map to `INTERNAL`

## Stub generation

```bash
make grpc-generate
make grpc-check
make test-grpc-parity
```

Generated files are created on demand and are not committed:

- `src/converter_grpc/converter_pb2.py`
- `src/converter_grpc/converter_pb2_grpc.py`

Stub generation uses pinned tool/runtime versions for reproducibility:

- `grpcio-tools==1.78.0`
- `grpcio==1.78.0`
- `protobuf==6.33.5`

Optional overrides:

- `GRPCIO_TOOLS_VERSION`
- `GRPCIO_VERSION`
- `PROTOBUF_VERSION`

Remove generated stubs:

```bash
rm -f src/converter_grpc/converter_pb2.py src/converter_grpc/converter_pb2_grpc.py
```

## Docker Compose smoke example

Run HTTP + gRPC services and a smoke client in one stack:

```bash
docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke
```
