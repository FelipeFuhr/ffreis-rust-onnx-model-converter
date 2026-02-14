"""Converter namespace with lazy imports for optional dependency isolation."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"


def convert_pytorch_to_onnx(
    model: Any,
    output_path: str,
    input_shape: tuple[int, ...],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    """Convert a PyTorch model to ONNX via lazy backend import.

    Parameters
    ----------
    model : Any
        PyTorch model object.
    output_path : str
        Destination ONNX path.
    input_shape : tuple[int, ...]
        Input tensor shape for export.
    input_names : list[str] | None, default=None
        Optional ONNX input names.
    output_names : list[str] | None, default=None
        Optional ONNX output names.
    dynamic_axes : dict[str, dict[int, str]] | None, default=None
        Dynamic axes mapping forwarded to the exporter.
    opset_version : int, default=14
        Target ONNX opset.
    **kwargs : Any
        Additional exporter-specific options.

    Returns
    -------
    str
        Path to the generated ONNX file.
    """
    from .pytorch_converter import convert_pytorch_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        input_shape=input_shape,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        **kwargs,
    )


def convert_tensorflow_to_onnx(
    model: Any,
    output_path: str,
    input_signature: Any = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    """Convert a TensorFlow/Keras model to ONNX via lazy backend import.

    Parameters
    ----------
    model : Any
        TensorFlow/Keras model object or compatible reference.
    output_path : str
        Destination ONNX path.
    input_signature : Any, default=None
        Optional TensorFlow input signature.
    opset_version : int, default=14
        Target ONNX opset.
    **kwargs : Any
        Additional converter-specific options.

    Returns
    -------
    str
        Path to the generated ONNX file.
    """
    from .tensorflow_converter import convert_tensorflow_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        input_signature=input_signature,
        opset_version=opset_version,
        **kwargs,
    )


def convert_sklearn_to_onnx(
    model: Any,
    output_path: str,
    initial_types: Any = None,
    target_opset: int | None = None,
    **kwargs: Any,
) -> str:
    """Convert a scikit-learn model to ONNX via lazy backend import.

    Parameters
    ----------
    model : Any
        Scikit-learn estimator or pipeline.
    output_path : str
        Destination ONNX path.
    initial_types : Any, default=None
        Optional type hints for ``skl2onnx`` conversion.
    target_opset : int | None, default=None
        Optional ONNX opset override.
    **kwargs : Any
        Additional converter-specific options.

    Returns
    -------
    str
        Path to the generated ONNX file.
    """
    from .sklearn_converter import convert_sklearn_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        initial_types=initial_types,
        target_opset=target_opset,
        **kwargs,
    )


__all__ = [
    "convert_pytorch_to_onnx",
    "convert_tensorflow_to_onnx",
    "convert_sklearn_to_onnx",
]
