# ONNX Model Converter

Convert model artifacts from PyTorch, TensorFlow/Keras, and scikit-learn into ONNX with optional parity checks, post-processing, and plugin-based extension points.

## What this project does

- Converts model artifacts to ONNX through Python API and CLI.
- Supports safer loading defaults for pickle-based formats.
- Supports optional parity checks between source model outputs and ONNX Runtime outputs.
- Supports optional ONNX post-processing (metadata, optimization, dynamic quantization).
- Supports plugin-based custom model families (for example AutoSklearn wrappers).

## Architecture (current)

The codebase follows a layered structure to separate orchestration from framework mechanics.

- `src/onnx_converter/application/`
- Use-cases and typed options (`ConversionOptions`, `ParityOptions`, `PostprocessOptions`).
- Orchestrates loaders, converters, parity checks, and post-processing.

- `src/onnx_converter/adapters/`
- Framework-specific adapter implementations.
- Model loaders, model-to-ONNX converters, and parity checkers.

- `src/onnx_converter/infrastructure/`
- ONNX post-processing implementation details.

- `src/onnx_converter/plugins/`
- Plugin protocol, registry, and built-in plugins.
- Entry point for custom model families that are not first-class in core.

- `src/onnx_converter/converters/`
- Lower-level framework conversion helpers.

- `src/onnx_converter/cli/`
- Typer CLI commands and dependency gates.

- `src/onnx_converter/api.py`
- Public file-based API wrappers over application use-cases.

For a deeper walkthrough (sequence, boundaries, plugin authoring), see `docs/architecture.md`.
For advanced plugin patterns and testing contract details, see `docs/plugin-authoring.md`.

## Request flow

1. CLI/API collects arguments and dependency checks.
2. API builds typed conversion options.
3. Application use-case validates config and orchestrates ports.
4. Adapter loader reads model artifact.
5. Adapter converter exports ONNX.
6. Optional parity check runs.
7. Optional post-process runs.
8. Output path is returned.

## Installation

### Minimal

```bash
pip install -e .
```

### Feature extras

```bash
pip install -e ".[cli]"
pip install -e ".[torch]"
pip install -e ".[tf_legacy]"   # TensorFlow/tf2onnx conversion path
pip install -e ".[tensorflow]"  # backward-compatible alias
pip install -e ".[sklearn]"
pip install -e ".[optuna]"
pip install -e ".[runtime]"
pip install -e ".[all]"         # secure default bundle (excludes tf_legacy)
```

## CLI usage

```bash
# PyTorch artifact -> ONNX
convert-to-onnx pytorch model.pt output.onnx --input-shape 1 3 224 224

# TensorFlow/Keras artifact -> ONNX
convert-to-onnx tensorflow saved_model_dir output.onnx

# sklearn artifact -> ONNX
convert-to-onnx sklearn model.joblib output.onnx --n-features 4 --allow-unsafe

# Plugin-based conversion
convert-to-onnx custom automl.joblib output.onnx \
  --model-type autosklearn \
  --plugin-module examples/autosklearn_plugin.py \
  --plugin-name autosklearn \
  --n-features 20 \
  --allow-unsafe
```

## Python API usage

```python
from pathlib import Path
from onnx_converter.api import convert_sklearn_file_to_onnx

convert_sklearn_file_to_onnx(
    model_path=Path("model.joblib"),
    output_path=Path("model.onnx"),
    n_features=8,
    allow_unsafe=True,
    optimize=True,
)
```

## Examples

All examples are in `examples/` and are designed to fail fast with clear criteria.

```bash
python examples/pytorch_example.py
python examples/tensorflow_example.py
python examples/sklearn_example.py
python examples/sklearn_custom_cli_example.py
python examples/compare_sklearn_vs_onnx.py
python examples/optuna_sklearn_example.py
python examples/autosklearn_roundtrip.py
```

## Developer workflow

Default local version is Python 3.13 (`Makefile` default). CI also defaults to 3.13, with matrix runs including older versions.

```bash
make env
make install-dev
make check
make coverage
make architecture-check
make ci-local
```

### Useful targets

- `make test-unit`: run fast unit-oriented suite (`-m "not integration"`)
- `make test-integration`: run integration-marked tests
- `make deps-sync-check`: verify `requirements.txt` is synced with `pyproject.toml`
- `make deps-sync-generate`: regenerate `requirements.txt` from `pyproject.toml`
- `make architecture-check`: boundary + complexity + strict mypy for application layer

## CI overview

- `build-python.yml`: matrix tests/package build (`3.13`, `3.11`, `3.10`)
- `lint.yml`: ruff + mypy
- `coverage.yml`: coverage report (threshold configured in `pyproject.toml`)
- `architecture.yml`: dependency sync + architecture + complexity checks
- `integration.yml`: scheduled/manual integration tests
- `examples.yml`: dockerized example runs (fast PR set + heavier scheduled set)

`tf_legacy` note:
- TensorFlow conversion is intentionally separated from `all` because current `tf2onnx` constraints on Python `<3.13` can pin older `protobuf`.
- CI/security gates target the secure default dependency profile; TensorFlow compatibility is tested in a dedicated integration job.

## Security and serialization notes

- Pickle-based artifact loading is unsafe for untrusted input.
- The project defaults to safer behavior and requires explicit `--allow-unsafe` for risky paths.
- Prefer safer formats when available (`.skops`, TorchScript/state_dict workflows, or direct in-memory export).

## Project map

- `src/onnx_converter/`: library code
- `tests/`: unit and integration tests
- `examples/`: executable example specs
- `container/`: example/runtime container definitions
- `scripts/`: CI/static architecture checks
- `.github/workflows/`: CI workflows

## Requirements

- Python `>=3.10` (project support)
- Python `3.13` (default local/CI baseline)

## License

MIT
