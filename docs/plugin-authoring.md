# Plugin Authoring Guide

This guide focuses on advanced plugin implementation patterns for custom model families.

## Security Considerations

**Important:** Plugin loading executes arbitrary Python code from the specified module or file path. This presents a security risk:

- **Only use plugins from trusted sources.** Malicious plugin code can access your file system, network, and execute arbitrary commands.
- **Never load plugins from untrusted or unknown sources.** This includes plugins from unverified repositories, unknown authors, or external/untrusted file paths.
- **Review plugin code before use.** Always inspect the source code of third-party plugins before loading them.
- **Plugin loading requires explicit user action.** The CLI requires explicit `--plugin-module` flags, ensuring users are aware when external code is being executed.

The plugin loading mechanism is designed for trusted development and deployment scenarios where the user has control over the plugin sources. It is not designed to safely execute untrusted code.

## When to Use a Plugin

Use a plugin when:

- the model family is not a first-class built-in command,
- serialization/loading requires custom behavior,
- export requires custom converter registration or graph logic.

Examples:

- AutoML wrappers (AutoSklearn-like artifacts),
- custom sklearn transformers/estimators,
- internal model wrappers around third-party frameworks.

## Plugin Contract

Implement the protocol from `src/onnx_converter/plugins/base.py`:

- `name: str`
- `can_handle(model_path, model_type, options) -> bool`
- `convert(model_path, output_path, options) -> Path`

Keep `can_handle` deterministic and cheap. Put expensive checks in `convert`.

## Option Validation (Required Pattern)

Always validate `options` with pydantic before doing work.

```python
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field


class MyOptions(BaseModel):
    n_features: int = Field(gt=0)
    allow_unsafe: bool = False


def parse_options(options: Mapping[str, Any]) -> MyOptions:
    return MyOptions.model_validate(dict(options))
```

This gives consistent errors and avoids implicit coercions.

## Loader Strategy

Recommended order:

1. Prefer safe formats and safe loaders.
2. Require explicit opt-in for unsafe deserialization (`allow_unsafe`).
3. Raise domain errors (`PluginError`, `UnsafeLoadError`, `UnsupportedModelError`) with actionable messages.

## Conversion Strategy

Recommended order:

1. Load model artifact.
2. Convert to ONNX with explicit/validated shape and opset options.
3. Run optional parity check.
4. Run post-processing (metadata, optimization, quantization).
5. Return output path.

Reuse existing adapters when possible:

- `SklearnModelLoader`, `TorchModelLoader`, etc.
- `SklearnModelConverter`, `TorchModelConverter`, etc.
- parity checkers and postprocessor implementation.

## Custom sklearn Converter Registration

If your model includes custom sklearn components:

1. implement `skl2onnx` shape calculator + converter function,
2. register via `update_registered_converter(...)`,
3. ensure registration occurs before conversion.

Pattern is shown in `examples/custom_sklearn_transformer.py`.

## Parity Strategy

Use parity as a contract, not a best-effort check.

- Validate label parity first (classification).
- Validate probability/tensor parity with clear tolerances.
- Fail with max absolute diff and tolerance values.

Recommended option names:

- `parity_input_path`
- `parity_atol`
- `parity_rtol`

## Plugin Registration Options

A plugin module can expose one of:

- `register_plugins(registry)` function,
- `PLUGINS` list,
- `PLUGIN` singleton.

CLI usage:

```bash
convert-to-onnx custom model.artifact out.onnx \
  --plugin-module path/to/my_plugin.py \
  --plugin-name my_plugin
```

## Testing Contract for Plugins

Each plugin should have tests covering:

1. option validation failures (type/shape/range),
2. deterministic `can_handle` behavior,
3. successful conversion path (mock heavy dependencies),
4. parity failure surfacing (clear errors),
5. postprocess integration (metadata/flags passed through).

For heavy end-to-end validation, add one small deterministic integration test and one dockerized example.

## CI Expectations

At minimum, plugin additions should pass:

- lint (`ruff`, `mypy`),
- unit tests,
- architecture checks,
- dependency sync checks.

If plugin adds new optional dependencies:

- add them to `pyproject.toml` extras,
- regenerate `requirements.txt` via `scripts/generate_requirements.py`,
- wire example docker image if needed.

## Design Rules for Maintainability

- Keep policy in application layer and mechanics in adapters/plugins.
- Do not print from conversion logic; return data/errors and let CLI format output.
- Use domain-specific errors with stable messages.
- Keep plugin options explicit and typed; avoid boolean-flag drift.

