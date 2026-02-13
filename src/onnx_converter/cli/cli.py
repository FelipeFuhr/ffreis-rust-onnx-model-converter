#!/usr/bin/env python3
"""
onnx_converter.cli.app

Typer-based CLI for converting models to ONNX with optional dependencies.

This CLI is designed to keep the core library lightweight while allowing users
to install only the framework extras they need.

Examples
--------
Install core + CLI only:

    uv pip install -e ".[cli]"

Install CLI + sklearn support:

    uv pip install -e ".[cli,sklearn]"

Install everything:

    uv pip install -e ".[cli,sklearn,torch,tensorflow,runtime]"
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from onnx_converter.errors import ConversionError, PluginError

app = typer.Typer(
    name="convert-to-onnx",
    help="Convert ML models (PyTorch / TensorFlow / sklearn) to ONNX.",
    no_args_is_help=True,
)

OPTIMIZE_PURPOSE = "ONNX graph optimization"
PARITY_INPUT_HELP = "Path to .npy/.npz/.csv/.txt input batch for parity check."
PARITY_RTOL_HELP = "Relative tolerance for parity check."
METADATA_HELP = "Custom ONNX metadata KEY=VALUE (repeatable)."
QUANTIZE_HELP = "Apply ONNX Runtime dynamic quantization."
OPTIMIZE_HELP = "Optimize ONNX graph after conversion."
QUANTIZE_PURPOSE = "dynamic quantization"


# -----------------------------
# Dependency checks / utilities
# -----------------------------
@dataclass(frozen=True)
class MissingDep:
    """Represent a missing optional dependency."""

    import_name: str
    extra_name: str
    purpose: str


def _is_importable(module: str) -> bool:
    """Check whether a module can be resolved.

    Parameters
    ----------
    module : str
        Module name to resolve.

    Returns
    -------
    bool
        ``True`` if the module can be imported, otherwise ``False``.
    """
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except Exception:
        # Extremely defensive: if spec resolution fails, treat as missing.
        return False


def _require_deps(missing: Sequence[MissingDep]) -> None:
    """Raise a Typer error if any required deps are missing.

    Parameters
    ----------
    missing : Sequence[MissingDep]
        Missing dependency requirements for a command.
    """
    not_found = [d for d in missing if not _is_importable(d.import_name)]
    if not not_found:
        return

    # Build a helpful message with uv extras
    extras = sorted({d.extra_name for d in not_found})
    details = "\n".join(
        f"- Missing '{d.import_name}' ({d.purpose}). Install extra: [bold].[{d.extra_name}][/bold]"
        for d in not_found
    )

    uv_hint = f'uv pip install -e ".[cli,{",".join(extras)}]"'
    pip_hint = f'pip install "onnx-model-converter[cli,{",".join(extras)}]"'

    msg = (
        "[red]Missing optional dependencies for this command.[/red]\n\n"
        f"{details}\n\n"
        "Install with uv (recommended):\n"
        f"  {uv_hint}\n\n"
        "Or with pip:\n"
        f"  {pip_hint}\n"
    )
    raise typer.BadParameter(msg)


def _import_custom_module(module_or_path: str) -> None:
    """Import a module by name or file path to register custom converters.

    Parameters
    ----------
    module_or_path : str
        Import path or filesystem path to a Python module.

    Raises
    ------
    typer.BadParameter
        If the module cannot be loaded or imported.
    """
    from onnx_converter.plugins.registry import _import_module_or_path

    try:
        _import_module_or_path(module_or_path)
    except PluginError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _print_conversion_error(exc: Exception, debug: bool) -> int:
    """Print a user-friendly conversion error.

    Parameters
    ----------
    exc : Exception
        Exception raised during conversion.
    debug : bool
        Whether to include traceback details.

    Returns
    -------
    int
        Process exit code.
    """
    typer.echo(f"[red]✗ {type(exc).__name__}:[/red] {exc}", err=True)
    if debug:
        typer.echo("\n[dim]Traceback:[/dim]", err=True)
        typer.echo("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), err=True)
    code = getattr(exc, "exit_code", None)
    if isinstance(code, int) and code > 0:
        return code
    return 1


def _parse_metadata(metadata_items: list[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE metadata entries."""
    parsed: dict[str, str] = {}
    for item in metadata_items or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid metadata entry '{item}'. Use KEY=VALUE format."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter("Metadata key cannot be empty.")
        parsed[key] = value
    return parsed


def _coerce_option_value(raw: str) -> object:
    """Best-effort coercion for CLI key/value options."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_model_options(option_items: list[str] | None) -> dict[str, object]:
    """Parse repeatable KEY=VALUE options for custom plugin command."""
    parsed: dict[str, object] = {}
    for item in option_items or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid option entry '{item}'. Use KEY=VALUE format."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter("Option key cannot be empty.")
        parsed[key] = _coerce_option_value(raw_value)
    return parsed


def _validate_if_requested(output_path: Path, validate: bool) -> None:
    """Validate ONNX output if requested.

    Parameters
    ----------
    output_path : Path
        Path to the generated ONNX file.
    validate : bool
        Whether runtime validation should be executed.

    Notes
    -----
    - Validation requires `onnx` (core) and `onnxruntime` (runtime extra).
    """
    if not validate:
        return

    _require_deps(
        [
            MissingDep("onnxruntime", "runtime", "runtime loading / inference validation"),
        ]
    )

    from onnx_converter.validate import validate_onnx_if_requested

    validate_onnx_if_requested(output_path, validate=True)


# -----------------------------
# Global options
# -----------------------------
@app.callback()
def _main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks on error."),
) -> None:
    """Initialize shared CLI state.

    Parameters
    ----------
    ctx : typer.Context
        Typer context object used to store shared state.
    debug : bool, default=False
        Whether to enable debug error output.
    """
    ctx.obj = {"debug": debug}


# -----------------------------
# Commands
# -----------------------------
@app.command("pytorch")
def pytorch_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a .pt/.pth model.",
    ),
    output_path: Path = typer.Argument(..., help="Where to write the .onnx file."),
    input_shape: list[int] = typer.Option(
        ...,
        "--input-shape",
        help="Input shape as repeated ints. Example: --input-shape 1 3 224 224",
    ),
    opset_version: int = typer.Option(14, "--opset-version", help="ONNX opset version."),
    input_names: list[str] | None = typer.Option(
        None, "--input-name", help="Input tensor name (repeatable)."
    ),
    output_names: list[str] | None = typer.Option(
        None, "--output-name", help="Output tensor name (repeatable)."
    ),
    dynamic_batch: bool = typer.Option(
        False, "--dynamic-batch", help="Mark batch axis (dim 0) as dynamic."
    ),
    allow_unsafe: bool = typer.Option(
        False,
        "--allow-unsafe",
        help="Allow unsafe pickle-based loading for PyTorch models (torch.load fallback).",
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help=OPTIMIZE_HELP,
    ),
    quantize_dynamic: bool = typer.Option(
        False, "--quantize-dynamic", help=QUANTIZE_HELP
    ),
    metadata: list[str] | None = typer.Option(
        None, "--metadata", help=METADATA_HELP
    ),
    parity_input: Path | None = typer.Option(
        None,
        "--parity-input",
        exists=True,
        readable=True,
        help=PARITY_INPUT_HELP,
    ),
) -> None:
    """Convert a PyTorch model to ONNX.

    Parameters
    ----------
    ctx : typer.Context
        Typer context containing global options.
    model_path : Path
        Path to a ``.pt`` or ``.pth`` model file.
    output_path : Path
        Destination path for the ONNX output.
    input_shape : list[int]
        Input tensor shape passed to the exporter.
    opset_version : int, default=14
        ONNX opset version used for export.
    allow_unsafe : bool, default=False
        Whether to allow pickle-based fallback loading.

    Notes
    -----
    - Requires the `torch` extra.
    - If your model file is TorchScript, your loader should prefer `torch.jit.load`.
    - Fallback first tries constrained `torch.load(..., weights_only=True)` when available.
    - Full `torch.load` deserialization is pickle-based and unsafe unless the file is trusted.
    """
    debug: bool = bool(ctx.obj.get("debug", False))

    _require_deps(
        [
            MissingDep("torch", "torch", "PyTorch model loading/export"),
        ]
    )
    if quantize_dynamic:
        _require_deps([MissingDep("onnxruntime", "runtime", QUANTIZE_PURPOSE)])
    if optimize:
        _require_deps([MissingDep("onnxoptimizer", "runtime", OPTIMIZE_PURPOSE)])

    metadata_payload = _parse_metadata(metadata)

    try:
        from onnx_converter.api import convert_torch_file_to_onnx

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "output_path": output_path,
            "input_shape": tuple(input_shape),
            "opset_version": opset_version,
            "allow_unsafe": allow_unsafe,
        }
        if input_names:
            kwargs["input_names"] = input_names
        if output_names:
            kwargs["output_names"] = output_names
        if dynamic_batch:
            kwargs["dynamic_batch"] = True
        if optimize:
            kwargs["optimize"] = True
        if quantize_dynamic:
            kwargs["quantize_dynamic"] = True
        if metadata_payload:
            kwargs["metadata"] = metadata_payload
        if parity_input is not None:
            kwargs["parity_input_path"] = parity_input

        out = convert_torch_file_to_onnx(
            **kwargs,
        )
        typer.echo(f"[green]✓ Saved:[/green] {out}")
    except ConversionError as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))
    except Exception as exc:
        # Unexpected crash: still show a clean message; debug prints traceback.
        raise typer.Exit(code=_print_conversion_error(exc, debug))


@app.command("tensorflow")
def tensorflow_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to a SavedModel directory or a Keras .h5 file.",
    ),
    output_path: Path = typer.Argument(..., help="Where to write the .onnx file."),
    opset_version: int = typer.Option(14, "--opset-version", help="ONNX opset version."),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help=OPTIMIZE_HELP,
    ),
    quantize_dynamic: bool = typer.Option(
        False, "--quantize-dynamic", help=QUANTIZE_HELP
    ),
    metadata: list[str] | None = typer.Option(
        None, "--metadata", help=METADATA_HELP
    ),
    parity_input: Path | None = typer.Option(
        None,
        "--parity-input",
        exists=True,
        readable=True,
        help=PARITY_INPUT_HELP,
    ),
    parity_atol: float = typer.Option(
        1e-5, "--parity-atol", help="Absolute tolerance for parity check."
    ),
    parity_rtol: float = typer.Option(
        1e-4, "--parity-rtol", help=PARITY_RTOL_HELP
    ),
    validate: bool = typer.Option(
        False, "--validate", help="Validate resulting ONNX with onnx + onnxruntime."
    ),
) -> None:
    """Convert a TensorFlow/Keras model to ONNX.

    Parameters
    ----------
    ctx : typer.Context
        Typer context containing global options.
    model_path : Path
        Path to a SavedModel directory or a Keras ``.h5`` file.
    output_path : Path
        Destination path for the ONNX output.
    opset_version : int, default=14
        ONNX opset version used for export.
    validate : bool, default=False
        Whether to validate the generated ONNX model.

    Notes
    -----
    - Requires the `tensorflow` extra (and typically `tf2onnx`).
    """
    debug: bool = bool(ctx.obj.get("debug", False))

    _require_deps(
        [
            MissingDep("tensorflow", "tensorflow", "TensorFlow/Keras model loading"),
            MissingDep("tf2onnx", "tensorflow", "TensorFlow → ONNX conversion"),
        ]
    )
    if quantize_dynamic:
        _require_deps([MissingDep("onnxruntime", "runtime", QUANTIZE_PURPOSE)])
    if optimize:
        _require_deps([MissingDep("onnxoptimizer", "runtime", OPTIMIZE_PURPOSE)])

    metadata_payload = _parse_metadata(metadata)

    try:
        from onnx_converter.api import convert_tf_path_to_onnx

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "output_path": output_path,
            "opset_version": opset_version,
        }
        if optimize:
            kwargs["optimize"] = True
        if quantize_dynamic:
            kwargs["quantize_dynamic"] = True
        if metadata_payload:
            kwargs["metadata"] = metadata_payload
        if parity_input is not None:
            kwargs["parity_input_path"] = parity_input
            kwargs["parity_atol"] = parity_atol
            kwargs["parity_rtol"] = parity_rtol

        out = convert_tf_path_to_onnx(**kwargs)
        _validate_if_requested(out, validate)
        typer.echo(f"[green]✓ Saved:[/green] {out}")
    except ConversionError as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))
    except Exception as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))


@app.command("sklearn")
def sklearn_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to .joblib/.skops/.pkl model.",
    ),
    output_path: Path = typer.Argument(..., help="Where to write the .onnx file."),
    n_features: int = typer.Option(..., "--n-features", min=1, help="Number of input features."),
    custom_converter_module: str | None = typer.Option(
        None,
        "--custom-converter-module",
        help="Python module or file path that registers custom skl2onnx converters.",
    ),
    allow_unsafe: bool = typer.Option(
        False,
        "--allow-unsafe",
        help="Allow unsafe pickle-based loading for sklearn (.joblib/.pkl). Prefer .skops.",
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help=OPTIMIZE_HELP,
    ),
    quantize_dynamic: bool = typer.Option(
        False, "--quantize-dynamic", help=QUANTIZE_HELP
    ),
    metadata: list[str] | None = typer.Option(
        None, "--metadata", help=METADATA_HELP
    ),
    parity_input: Path | None = typer.Option(
        None,
        "--parity-input",
        exists=True,
        readable=True,
        help="Path to .npy/.npz/.csv/.txt feature matrix for parity check.",
    ),
    parity_atol: float = typer.Option(
        1e-5, "--parity-atol", help="Absolute tolerance for parity check."
    ),
    parity_rtol: float = typer.Option(
        1e-4, "--parity-rtol", help=PARITY_RTOL_HELP
    ),
    validate: bool = typer.Option(
        False, "--validate", help="Validate resulting ONNX with onnx + onnxruntime."
    ),
) -> None:
    """Convert a Scikit-learn model to ONNX.

    Parameters
    ----------
    ctx : typer.Context
        Typer context containing global options.
    model_path : Path
        Path to a serialized scikit-learn model artifact.
    output_path : Path
        Destination path for the ONNX output.
    n_features : int
        Number of model input features.
    custom_converter_module : str | None, default=None
        Optional module path for custom ``skl2onnx`` converter registration.
    allow_unsafe : bool, default=False
        Whether to allow loading pickle-based ``.pkl`` artifacts.
    validate : bool, default=False
        Whether to validate the generated ONNX model.

    Notes
    -----
    - Requires the `sklearn` extra (scikit-learn + skl2onnx + joblib + skops).
    - `.skops` is preferred for safer loading when available.
    """
    debug: bool = bool(ctx.obj.get("debug", False))

    _require_deps(
        [
            MissingDep("sklearn", "sklearn", "Scikit-learn model support"),
            MissingDep("skl2onnx", "sklearn", "Sklearn → ONNX conversion"),
            MissingDep("joblib", "sklearn", "Joblib model loading"),
            MissingDep("skops", "sklearn", "Safer sklearn serialization (.skops)"),
        ]
    )
    if quantize_dynamic:
        _require_deps([MissingDep("onnxruntime", "runtime", QUANTIZE_PURPOSE)])
    if optimize:
        _require_deps([MissingDep("onnxoptimizer", "runtime", OPTIMIZE_PURPOSE)])

    if custom_converter_module:
        _import_custom_module(custom_converter_module)

    metadata_payload = _parse_metadata(metadata)

    try:
        from onnx_converter.api import convert_sklearn_file_to_onnx

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "output_path": output_path,
            "n_features": n_features,
            "allow_unsafe": allow_unsafe,
        }
        if optimize:
            kwargs["optimize"] = True
        if quantize_dynamic:
            kwargs["quantize_dynamic"] = True
        if metadata_payload:
            kwargs["metadata"] = metadata_payload
        if parity_input is not None:
            kwargs["parity_input_path"] = parity_input
            kwargs["parity_atol"] = parity_atol
            kwargs["parity_rtol"] = parity_rtol

        out = convert_sklearn_file_to_onnx(**kwargs)
        _validate_if_requested(out, validate)
        typer.echo(f"[green]✓ Saved:[/green] {out}")
    except ConversionError as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))
    except Exception as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))


@app.command("custom")
def custom_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to model artifact handled by a plugin.",
    ),
    output_path: Path = typer.Argument(..., help="Where to write the .onnx file."),
    model_type: str | None = typer.Option(
        None,
        "--model-type",
        help="Optional model family hint (e.g. autosklearn).",
    ),
    plugin_name: str | None = typer.Option(
        None, "--plugin-name", help="Explicit plugin name."
    ),
    plugin_module: list[str] | None = typer.Option(
        None,
        "--plugin-module",
        help="Plugin module import path or file path (repeatable).",
    ),
    n_features: int | None = typer.Option(
        None, "--n-features", help="Input features (used by sklearn-like plugins)."
    ),
    allow_unsafe: bool = typer.Option(
        False, "--allow-unsafe", help="Allow unsafe pickle-based model loading."
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help=OPTIMIZE_HELP,
    ),
    quantize_dynamic: bool = typer.Option(
        False, "--quantize-dynamic", help=QUANTIZE_HELP
    ),
    metadata: list[str] | None = typer.Option(
        None, "--metadata", help=METADATA_HELP
    ),
    parity_input: Path | None = typer.Option(
        None,
        "--parity-input",
        exists=True,
        readable=True,
        help=PARITY_INPUT_HELP,
    ),
    option: list[str] | None = typer.Option(
        None,
        "--option",
        help="Plugin option KEY=VALUE (repeatable).",
    ),
) -> None:
    """Convert model using plugin-based adapter resolution."""
    debug: bool = bool(ctx.obj.get("debug", False))

    metadata_payload = _parse_metadata(metadata)
    option_payload = _parse_model_options(option)
    if n_features is not None:
        option_payload["n_features"] = n_features
    option_payload["allow_unsafe"] = allow_unsafe
    option_payload["optimize"] = optimize
    option_payload["quantize_dynamic"] = quantize_dynamic
    option_payload["metadata"] = metadata_payload
    if parity_input is not None:
        option_payload["parity_input_path"] = parity_input

    try:
        from onnx_converter.api import convert_custom_file_to_onnx

        out = convert_custom_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            model_type=model_type,
            plugin_name=plugin_name,
            plugin_modules=plugin_module,
            options=option_payload,
        )
        typer.echo(f"[green]✓ Saved:[/green] {out}")
    except ConversionError as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))
    except Exception as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))


@app.command("doctor")
def doctor_cmd() -> None:
    """Print installed toolchain versions and compatibility notes."""
    import importlib.metadata as metadata

    modules = [
        "onnx",
        "onnxruntime",
        "onnxoptimizer",
        "torch",
        "tensorflow",
        "tf2onnx",
        "scikit-learn",
        "skl2onnx",
    ]

    typer.echo(f"Python: {sys.version.split()[0]}")
    for module in modules:
        try:
            version = metadata.version(module)
            typer.echo(f"{module}: {version}")
        except metadata.PackageNotFoundError:
            typer.echo(f"{module}: <not installed>")

    py_ver = tuple(sys.version_info[:2])
    try:
        tf_version = metadata.version("tensorflow")
    except metadata.PackageNotFoundError:
        tf_version = None

    if py_ver >= (3, 12) and tf_version is None:
        typer.echo(
            "[yellow]Note:[/yellow] TensorFlow wheels may require Python 3.11 for some versions."
        )

    try:
        from onnx_converter.plugins.registry import create_default_registry

        registry = create_default_registry()
        typer.echo(f"plugins: {', '.join(registry.names())}")
    except Exception:
        typer.echo("plugins: <unavailable>")


if __name__ == "__main__":
    app()
