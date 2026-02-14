"""Application use-cases orchestrating conversion workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import ValidationError

from onnx_converter.adapters.converters import (
    SklearnModelConverter,
    TensorflowModelConverter,
    TorchModelConverter,
)
from onnx_converter.adapters.loaders import (
    SklearnModelLoader,
    TensorflowModelLoader,
    TorchModelLoader,
)
from onnx_converter.adapters.parity_checkers import (
    SklearnParityChecker,
    TensorflowParityChecker,
    TorchParityChecker,
)
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
from onnx_converter.errors import ConversionError
from onnx_converter.infrastructure.postprocessing import OnnxPostProcessorImpl
from onnx_converter.plugins.registry import create_default_registry
from onnx_converter.schemas import (
    SklearnFileConversionConfig,
    TensorflowFileConversionConfig,
    TorchFileConversionConfig,
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
    """Use-case: convert PyTorch artifact into ONNX."""
    try:
        config = TorchFileConversionConfig(
            model_path=model_path,
            output_path=output_path,
            input_shape=tuple(input_shape),
            opset_version=options.opset_version,
            allow_unsafe=options.allow_unsafe,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid PyTorch conversion parameters: {exc}") from exc

    loader = loader or TorchModelLoader()
    converter = converter or TorchModelConverter()
    parity_checker = parity_checker or TorchParityChecker()
    postprocessor = postprocessor or OnnxPostProcessorImpl()

    model = loader.load(config.model_path, allow_unsafe=config.allow_unsafe)
    dynamic_axes = (
        {
            name: {0: "batch"}
            for name in (input_names or ["input"]) + (output_names or ["output"])
        }
        if dynamic_batch
        else None
    )
    out_path = converter.convert(
        model,
        config.output_path,
        options={
            "input_shape": config.input_shape,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "opset_version": config.opset_version,
        },
    )

    parity_checker.check(model, out_path, options.parity)
    postprocessor.run(
        output_path=out_path,
        source_path=config.model_path,
        framework="pytorch",
        config_metadata={
            "onnx_converter.opset": str(config.opset_version),
            "onnx_converter.input_shape": str(config.input_shape),
        },
        options=options.postprocess,
    )
    return ConversionResult(
        output_path=out_path,
        framework="pytorch",
        source_path=config.model_path,
        metadata=options.postprocess.metadata,
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
    """Use-case: convert TensorFlow/Keras artifact into ONNX."""
    try:
        config = TensorflowFileConversionConfig(
            model_path=model_path,
            output_path=output_path,
            opset_version=options.opset_version,
        )
    except ValidationError as exc:
        raise ConversionError(
            f"Invalid TensorFlow conversion parameters: {exc}"
        ) from exc

    loader = loader or TensorflowModelLoader()
    converter = converter or TensorflowModelConverter()
    parity_checker = parity_checker or TensorflowParityChecker()
    postprocessor = postprocessor or OnnxPostProcessorImpl()

    model = loader.load(config.model_path)
    out_path = converter.convert(
        model,
        config.output_path,
        options={"opset_version": config.opset_version},
    )
    parity_checker.check(model, out_path, options.parity)
    postprocessor.run(
        output_path=out_path,
        source_path=config.model_path,
        framework="tensorflow",
        config_metadata={"onnx_converter.opset": str(config.opset_version)},
        options=options.postprocess,
    )
    return ConversionResult(
        output_path=out_path,
        framework="tensorflow",
        source_path=config.model_path,
        metadata=options.postprocess.metadata,
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
    """Use-case: convert sklearn artifact into ONNX."""
    try:
        config = SklearnFileConversionConfig(
            model_path=model_path,
            output_path=output_path,
            n_features=n_features,
            allow_unsafe=options.allow_unsafe,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid sklearn conversion parameters: {exc}") from exc

    loader = loader or SklearnModelLoader()
    converter = converter or SklearnModelConverter()
    parity_checker = parity_checker or SklearnParityChecker()
    postprocessor = postprocessor or OnnxPostProcessorImpl()

    model = loader.load(config.model_path, allow_unsafe=config.allow_unsafe)
    out_path = converter.convert(
        model,
        config.output_path,
        options={
            "n_features": config.n_features,
            "target_opset": options.opset_version,
        },
    )
    parity_checker.check(model, out_path, options.parity)
    postprocessor.run(
        output_path=out_path,
        source_path=config.model_path,
        framework="sklearn",
        config_metadata={"onnx_converter.n_features": str(config.n_features)},
        options=options.postprocess,
    )
    return ConversionResult(
        output_path=out_path,
        framework="sklearn",
        source_path=config.model_path,
        metadata=options.postprocess.metadata,
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
    """Use-case: resolve and run conversion plugin."""
    option_map = dict(options)
    registry = create_default_registry(extra_modules=plugin_modules)
    plugin = registry.resolve(
        model_path=model_path,
        model_type=model_type,
        plugin_name=plugin_name,
        options=option_map,
    )
    out_path = plugin.convert(
        model_path=model_path,
        output_path=output_path,
        options=option_map,
    )
    return ConversionResult(
        output_path=out_path,
        framework=f"plugin:{plugin.name}",
        source_path=model_path,
        metadata=None,
    )


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
    """Build typed option object from command/API params."""
    return ConversionOptions(
        allow_unsafe=allow_unsafe,
        opset_version=opset_version,
        parity=ParityOptions(
            input_path=parity_input_path,
            atol=parity_atol,
            rtol=parity_rtol,
        ),
        postprocess=PostprocessOptions(
            optimize=optimize,
            quantize_dynamic=quantize_dynamic,
            metadata=metadata,
        ),
    )
