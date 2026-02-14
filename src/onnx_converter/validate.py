"""ONNX validation helpers."""

from __future__ import annotations

from pathlib import Path

from onnx_converter.errors import ConversionError


def validate_onnx_if_requested(output_path: Path, validate: bool) -> None:
    """Validate an ONNX model when validation is enabled.

    Parameters
    ----------
    output_path : Path
        Path to the ONNX model file.
    validate : bool
        Whether validation should be executed.

    Raises
    ------
    ConversionError
        If required validation dependencies are missing or validation fails.
    """
    if not validate:
        return

    try:
        import onnx
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - dependency guarded by CLI
        raise ConversionError(
            "Validation requires onnxruntime to be installed."
        ) from exc

    try:
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        ort.InferenceSession(str(output_path))
    except Exception as exc:
        raise ConversionError(f"ONNX validation failed: {exc}") from exc
