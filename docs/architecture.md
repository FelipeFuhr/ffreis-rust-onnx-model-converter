# Architecture Guide

This document describes how requests flow through the project, where each concern lives, and how to add custom model-family support safely.

## Layers

- `cli/`: user input, dependency checks, error presentation, exit codes.
- `api.py`: public file-based API wrappers.
- `application/`: use-cases and orchestration.
- `adapters/`: framework-specific loaders/converters/parity checkers.
- `infrastructure/`: ONNX post-processing implementation details.
- `plugins/`: plugin protocol, registry, and built-in plugin(s).
- `converters/`: lower-level framework conversion helpers.

## Dependency Direction

Allowed direction is inward:

`cli -> api -> application -> adapters -> converters/infrastructure`

`plugins` can use `adapters` and `application` option objects, but should not pull CLI concerns.

## Conversion Sequence

```text
User (CLI/API)
  -> parse/validate arguments
  -> build ConversionOptions
  -> application.use_case(...)
      -> loader.load(model_path)
      -> converter.convert(model, output_path, options)
      -> parity_checker.check(...)        # optional
      -> postprocessor.run(...)           # optional
  -> ConversionResult(output_path, ...)
```

## Core Option Objects

- `ConversionOptions`: top-level options.
- `ParityOptions`: parity input path and tolerances.
- `PostprocessOptions`: optimize, quantize, metadata.

These are passed through use-cases to avoid long primitive argument lists.

## Validation Strategy

- File-level request validation: `schemas.py` pydantic models (`*FileConversionConfig`).
- Export/plugin option validation: pydantic models (`PytorchConversionConfig`, `SklearnConversionConfig`, `SklearnPluginOptions`, `PluginResolutionConfig`).
- Domain-specific failures: `errors.py` hierarchy (`DependencyError`, `UnsafeLoadError`, `UnsupportedModelError`, `ParityError`, `PostprocessError`, `PluginError`).

## Plugin Model

Plugin contract is defined in `plugins/base.py`:

- `name: str`
- `can_handle(model_path, model_type, options) -> bool`
- `convert(model_path, output_path, options) -> Path`

Registry behavior in `plugins/registry.py`:

- explicit plugin (`--plugin-name`) wins.
- else resolve by `can_handle`.
- error on no match or multiple matches.

## Adding a New Plugin

1. Create a class implementing the plugin protocol.
2. Validate incoming options with a pydantic model.
3. Reuse existing adapters where possible (`ModelLoader`, `ModelConverter`, parity checker, postprocessor).
4. Register plugin using one of:
- module-level `register_plugins(registry)` function
- module-level `PLUGINS = [...]`
- module-level `PLUGIN = ...`
5. Load module through CLI `--plugin-module` or `create_default_registry(extra_modules=...)`.

For advanced patterns and testing expectations, see `docs/plugin-authoring.md`.

## Minimal Plugin Skeleton

```python
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field


class MyPluginOptions(BaseModel):
    n_features: int = Field(gt=0)


class MyPlugin:
    name = "my_plugin"

    def can_handle(
        self,
        model_path: Path,
        model_type: str | None,
        options: Mapping[str, Any],
    ) -> bool:
        return model_type == "my_family" or model_path.suffix == ".myfmt"

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        options: Mapping[str, Any],
    ) -> Path:
        parsed = MyPluginOptions.model_validate(dict(options))
        # load -> convert -> optional parity -> postprocess
        return output_path
```

## CI Enforcement

- `scripts/check_architecture.py`: import boundary checks.
- `scripts/check_orchestrator_complexity.py`: use-case complexity guard.

These are executed in `architecture.yml`.
