"""Public file-based conversion API (delegates to application use-cases)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from typing import Mapping
from typing import Optional

from onnx_converter.application.use_cases import build_conversion_options
from onnx_converter.application.use_cases import convert_custom_file
from onnx_converter.application.use_cases import convert_sklearn_file
from onnx_converter.application.use_cases import convert_tensorflow_file
from onnx_converter.application.use_cases import convert_torch_file


def convert_torch_file_to_onnx(
    model_path: Path,
    output_path: Path,
    input_shape: Iterable[int],
    opset_version: int = 14,
    allow_unsafe: bool = False,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_batch: bool = False,
    optimize: bool = False,
    quantize_dynamic: bool = False,
    metadata: Optional[Mapping[str, str]] = None,
    parity_input_path: Optional[Path] = None,
    parity_atol: float = 1e-5,
) -> Path:
    """Convert a serialized PyTorch model artifact to ONNX."""
    options = build_conversion_options(
        allow_unsafe=allow_unsafe,
        opset_version=opset_version,
        optimize=optimize,
        quantize_dynamic=quantize_dynamic,
        metadata=metadata,
        parity_input_path=parity_input_path,
        parity_atol=parity_atol,
    )
    result = convert_torch_file(
        model_path=model_path,
        output_path=output_path,
        input_shape=input_shape,
        options=options,
        input_names=input_names,
        output_names=output_names,
        dynamic_batch=dynamic_batch,
    )
    return result.output_path


def convert_tf_path_to_onnx(
    model_path: Path,
    output_path: Path,
    opset_version: int = 14,
    optimize: bool = False,
    quantize_dynamic: bool = False,
    metadata: Optional[Mapping[str, str]] = None,
    parity_input_path: Optional[Path] = None,
    parity_atol: float = 1e-5,
    parity_rtol: float = 1e-4,
) -> Path:
    """Convert a TensorFlow/Keras artifact to ONNX."""
    options = build_conversion_options(
        opset_version=opset_version,
        optimize=optimize,
        quantize_dynamic=quantize_dynamic,
        metadata=metadata,
        parity_input_path=parity_input_path,
        parity_atol=parity_atol,
        parity_rtol=parity_rtol,
    )
    result = convert_tensorflow_file(
        model_path=model_path,
        output_path=output_path,
        options=options,
    )
    return result.output_path


def convert_sklearn_file_to_onnx(
    model_path: Path,
    output_path: Path,
    n_features: int,
    allow_unsafe: bool = False,
    optimize: bool = False,
    quantize_dynamic: bool = False,
    metadata: Optional[Mapping[str, str]] = None,
    parity_input_path: Optional[Path] = None,
    parity_atol: float = 1e-5,
    parity_rtol: float = 1e-4,
) -> Path:
    """Convert a serialized sklearn artifact to ONNX."""
    options = build_conversion_options(
        allow_unsafe=allow_unsafe,
        optimize=optimize,
        quantize_dynamic=quantize_dynamic,
        metadata=metadata,
        parity_input_path=parity_input_path,
        parity_atol=parity_atol,
        parity_rtol=parity_rtol,
    )
    result = convert_sklearn_file(
        model_path=model_path,
        output_path=output_path,
        n_features=n_features,
        options=options,
    )
    return result.output_path


def convert_custom_file_to_onnx(
    model_path: Path,
    output_path: Path,
    *,
    model_type: Optional[str] = None,
    plugin_name: Optional[str] = None,
    plugin_modules: Optional[Iterable[str]] = None,
    options: Optional[Mapping[str, object]] = None,
) -> Path:
    """Convert model artifact via plugin-based adapter resolution."""
    result = convert_custom_file(
        model_path=model_path,
        output_path=output_path,
        model_type=model_type,
        plugin_name=plugin_name,
        plugin_modules=plugin_modules,
        options=options or {},
    )
    return result.output_path
