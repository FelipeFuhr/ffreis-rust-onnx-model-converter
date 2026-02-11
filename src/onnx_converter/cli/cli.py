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

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import typer

from onnx_converter.errors import ConversionError


app = typer.Typer(
    name="convert-to-onnx",
    help="Convert ML models (PyTorch / TensorFlow / sklearn) to ONNX.",
    no_args_is_help=True,
)


# -----------------------------
# Dependency checks / utilities
# -----------------------------
@dataclass(frozen=True)
class MissingDep:
    """Represents a missing optional dependency."""

    import_name: str
    extra_name: str
    purpose: str


def _is_importable(module: str) -> bool:
    """Return True if `module` can be imported (without importing it)."""
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
    missing:
        A list of MissingDep requirements for a given command.
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


def _print_conversion_error(exc: Exception, debug: bool) -> int:
    """Print a user-friendly error, optionally including tracebacks."""
    typer.echo(f"[red]✗ {type(exc).__name__}:[/red] {exc}", err=True)
    if debug:
        typer.echo("\n[dim]Traceback:[/dim]", err=True)
        typer.echo("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), err=True)
    return 1


def _validate_if_requested(output_path: Path, validate: bool) -> None:
    """Validate ONNX output if requested.

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
    """Entry point for shared CLI state."""
    ctx.obj = {"debug": debug}


# -----------------------------
# Commands
# -----------------------------
@app.command("pytorch")
def pytorch_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to a .pt/.pth model."),
    output_path: Path = typer.Argument(..., help="Where to write the .onnx file."),
    input_shape: list[int] = typer.Option(
        ...,
        "--input-shape",
        help="Input shape as repeated ints. Example: --input-shape 1 3 224 224",
    ),
    opset_version: int = typer.Option(14, "--opset-version", help="ONNX opset version."),
    allow_unsafe: bool = typer.Option(
        False,
        "--allow-unsafe",
        help="Allow unsafe pickle-based loading for PyTorch models (torch.load fallback).",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate resulting ONNX with onnx + onnxruntime.",
    ),
) -> None:
    """Convert a PyTorch model to ONNX.

    Notes
    -----
    - Requires the `torch` extra.
    - If your model file is TorchScript, your loader should prefer `torch.jit.load`.
    - If it falls back to `torch.load`, that is pickle-based and unsafe unless trusted.
    """
    debug: bool = bool(ctx.obj.get("debug", False))

    _require_deps(
        [
            MissingDep("torch", "torch", "PyTorch model loading/export"),
        ]
    )

    try:
        from onnx_converter.api import convert_torch_file_to_onnx

        out = convert_torch_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_shape=tuple(input_shape),
            opset_version=opset_version,
            allow_unsafe=allow_unsafe,
        )
        _validate_if_requested(out, validate)
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
    validate: bool = typer.Option(False, "--validate", help="Validate resulting ONNX with onnx + onnxruntime."),
) -> None:
    """Convert a TensorFlow/Keras model to ONNX.

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

    try:
        from onnx_converter.api import convert_tf_path_to_onnx

        out = convert_tf_path_to_onnx(
            model_path=model_path,
            output_path=output_path,
            opset_version=opset_version,
        )
        _validate_if_requested(out, validate)
        typer.echo(f"[green]✓ Saved:[/green] {out}")
    except ConversionError as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))
    except Exception as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))


@app.command("sklearn")
def sklearn_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to .joblib/.skops/.pkl model."),
    output_path: Path = typer.Argument(..., help="Where to write the .onnx file."),
    n_features: int = typer.Option(..., "--n-features", min=1, help="Number of input features."),
    allow_unsafe: bool = typer.Option(
        False,
        "--allow-unsafe",
        help="Allow unsafe pickle-based loading for sklearn (.pkl). Prefer .joblib or .skops.",
    ),
    validate: bool = typer.Option(False, "--validate", help="Validate resulting ONNX with onnx + onnxruntime."),
) -> None:
    """Convert a Scikit-learn model to ONNX.

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

    try:
        from onnx_converter.api import convert_sklearn_file_to_onnx

        out = convert_sklearn_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            n_features=n_features,
            allow_unsafe=allow_unsafe,
        )
        _validate_if_requested(out, validate)
        typer.echo(f"[green]✓ Saved:[/green] {out}")
    except ConversionError as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))
    except Exception as exc:
        raise typer.Exit(code=_print_conversion_error(exc, debug))


if __name__ == "__main__":
    app()
