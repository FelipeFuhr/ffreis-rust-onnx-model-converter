"""Top-level API for model-to-ONNX conversion."""

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
    """Convert a PyTorch model to ONNX.

    Parameters
    ----------
    model
        PyTorch model instance to convert.
    output_path : str
        Destination path for the ONNX output.
    input_shape : tuple[int, ...]
        Input tensor shape used for export.
    input_names : list, optional
        Input tensor names for the ONNX graph.
    output_names : list, optional
        Output tensor names for the ONNX graph.
    dynamic_axes : dict, optional
        Dynamic axis mapping forwarded to the exporter.
    opset_version : int, default=14
        ONNX opset version to use.
    **kwargs : Any
        Additional exporter options.

    Returns
    -------
    str
        Path to the generated ONNX file.
    """
    from .converters.pytorch_converter import convert_pytorch_to_onnx as _impl

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
    """Convert a TensorFlow/Keras model to ONNX.

    Parameters
    ----------
    model
        TensorFlow/Keras model instance or SavedModel path.
    output_path : str
        Destination path for the ONNX output.
    input_signature, optional
        TensorFlow input signature for conversion.
    opset_version : int, default=14
        ONNX opset version to use.
    **kwargs : Any
        Additional converter options.

    Returns
    -------
    str
        Path to the generated ONNX file.
    """
    from .converters.tensorflow_converter import convert_tensorflow_to_onnx as _impl

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
    """Convert a scikit-learn model or pipeline to ONNX.

    Parameters
    ----------
    model
        Scikit-learn model or pipeline instance.
    output_path : str
        Destination path for the ONNX output.
    initial_types, optional
        Input type declarations passed to ``skl2onnx``.
    target_opset : int, optional
        ONNX opset version override.
    **kwargs : Any
        Additional converter options.

    Returns
    -------
    str
        Path to the generated ONNX file.
    """
    from .converters.sklearn_converter import convert_sklearn_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        initial_types=initial_types,
        target_opset=target_opset,
        **kwargs,
    )


def convert_custom_file_to_onnx(**kwargs: Any) -> Any:
    """Convert model artifact through plugin-based adapter resolution."""
    from .api import convert_custom_file_to_onnx as _impl

    return _impl(**kwargs)


__all__ = [
    "convert_pytorch_to_onnx",
    "convert_tensorflow_to_onnx",
    "convert_sklearn_to_onnx",
    "convert_custom_file_to_onnx",
]
