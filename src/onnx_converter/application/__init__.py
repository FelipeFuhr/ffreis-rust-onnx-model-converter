"""Application-layer use-cases and option objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from onnx_converter.application.options import (
    ConversionOptions,
    ParityOptions,
    PostprocessOptions,
)
from onnx_converter.application.ports import (
    ModelConverter,
    ModelLoader,
    OnnxPostProcessor,
    ParityChecker,
)
from onnx_converter.application.results import ConversionResult


def build_conversion_options(
    *,
    allow_unsafe: bool = False,
    opset_version: int = 14,
    optimize: bool = False,
    quantize_dynamic: bool = False,
    metadata: Mapping[str, str] | None = None,
    parity_input_path: Path | None = None,
    parity_atol: float = 1e-5,
    parity_rtol: float = 1e-4,
) -> ConversionOptions:
    """Build typed conversion options via lazy use-case import."""
    from onnx_converter.application.use_cases import build_conversion_options as _impl

    return _impl(
        allow_unsafe=allow_unsafe,
        opset_version=opset_version,
        optimize=optimize,
        quantize_dynamic=quantize_dynamic,
        metadata=metadata,
        parity_input_path=parity_input_path,
        parity_atol=parity_atol,
        parity_rtol=parity_rtol,
    )


def convert_torch_file(
    *,
    model_path: Path,
    output_path: Path,
    input_shape: Iterable[int],
    options: ConversionOptions,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_batch: bool = False,
    loader: ModelLoader | None = None,
    converter: ModelConverter | None = None,
    parity_checker: ParityChecker | None = None,
    postprocessor: OnnxPostProcessor | None = None,
) -> ConversionResult:
    """Convert PyTorch model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_torch_file as _impl

    return _impl(
        model_path=model_path,
        output_path=output_path,
        input_shape=input_shape,
        options=options,
        input_names=input_names,
        output_names=output_names,
        dynamic_batch=dynamic_batch,
        loader=loader,
        converter=converter,
        parity_checker=parity_checker,
        postprocessor=postprocessor,
    )


def convert_tensorflow_file(
    *,
    model_path: Path,
    output_path: Path,
    options: ConversionOptions,
    loader: ModelLoader | None = None,
    converter: ModelConverter | None = None,
    parity_checker: ParityChecker | None = None,
    postprocessor: OnnxPostProcessor | None = None,
) -> ConversionResult:
    """Convert TensorFlow model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_tensorflow_file as _impl

    return _impl(
        model_path=model_path,
        output_path=output_path,
        options=options,
        loader=loader,
        converter=converter,
        parity_checker=parity_checker,
        postprocessor=postprocessor,
    )


def convert_sklearn_file(
    *,
    model_path: Path,
    output_path: Path,
    n_features: int,
    options: ConversionOptions,
    loader: ModelLoader | None = None,
    converter: ModelConverter | None = None,
    parity_checker: ParityChecker | None = None,
    postprocessor: OnnxPostProcessor | None = None,
) -> ConversionResult:
    """Convert sklearn model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_sklearn_file as _impl

    return _impl(
        model_path=model_path,
        output_path=output_path,
        n_features=n_features,
        options=options,
        loader=loader,
        converter=converter,
        parity_checker=parity_checker,
        postprocessor=postprocessor,
    )


def convert_custom_file(
    *,
    model_path: Path,
    output_path: Path,
    model_type: str | None,
    plugin_name: str | None,
    plugin_modules: Iterable[str] | None,
    options: Mapping[str, object],
) -> ConversionResult:
    """Convert custom model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_custom_file as _impl

    return _impl(
        model_path=model_path,
        output_path=output_path,
        model_type=model_type,
        plugin_name=plugin_name,
        plugin_modules=plugin_modules,
        options=options,
    )


__all__ = [
    "ConversionOptions",
    "ParityOptions",
    "PostprocessOptions",
    "ConversionResult",
    "build_conversion_options",
    "convert_torch_file",
    "convert_tensorflow_file",
    "convert_sklearn_file",
    "convert_custom_file",
]
