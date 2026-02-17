# Converter Transport API

`converter` accepts model artifacts over HTTP or gRPC, verifies optional
input SHA-256 integrity, converts to ONNX, and returns ONNX bytes plus output
integrity metadata.

## Install extras

```bash
pip install -e ".[server,grpc]"
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
    - `allow_unsafe` (optional, default `false`)

Response:

- body: ONNX bytes (`application/octet-stream`)
- headers:
  - `X-Input-SHA256`
  - `X-Output-SHA256`
  - `X-Output-Filename`

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

## Stub generation

```bash
make grpc-generate
make grpc-check
make test-grpc-parity
```

## Docker Compose smoke example

Run HTTP + gRPC services and a smoke client in one stack:

```bash
docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke
```
