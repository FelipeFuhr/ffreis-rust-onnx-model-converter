"""Built-in conversion plugins."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from onnx_converter.adapters.converters import SklearnModelConverter
from onnx_converter.adapters.loaders import SklearnModelLoader
from onnx_converter.adapters.parity_checkers import SklearnParityChecker
from onnx_converter.application.options import ParityOptions, PostprocessOptions
from onnx_converter.errors import PluginError
from onnx_converter.infrastructure.postprocessing import OnnxPostProcessorImpl
from onnx_converter.schemas import SklearnPluginOptions


class SklearnFilePlugin:
    """Convert serialized sklearn artifacts into ONNX.

    Notes
    -----
    Options are validated through ``SklearnPluginOptions`` before conversion.
    """

    name = "sklearn_file"

    def can_handle(
        self,
        model_path: Path,
        model_type: str | None,
    ) -> bool:
        """Check whether this plugin can convert the given artifact.

        Parameters
        ----------
        model_path : Path
            Path to the model artifact.
        model_type : str | None
            Optional model family hint.

        Returns
        -------
        bool
            ``True`` when this plugin should handle the model.
        """
        if model_type and model_type.lower() in {"sklearn", "autosklearn"}:
            return True
        return model_path.suffix.lower() in {
            ".skops",
            ".joblib",
            ".jl",
            ".pkl",
            ".pickle",
        }

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        options: Mapping[str, Any],
    ) -> Path:
        """Convert sklearn artifact to ONNX using validated options.

        Parameters
        ----------
        model_path : Path
            Input model artifact path.
        output_path : Path
            ONNX output path.
        options : Mapping[str, Any]
            User-provided plugin options.

        Returns
        -------
        Path
            Path to the generated ONNX model.

        Raises
        ------
        PluginError
            If plugin options are invalid or conversion fails.
        """
        try:
            parsed = SklearnPluginOptions.model_validate(dict(options))
        except ValidationError as exc:
            raise PluginError(f"Invalid sklearn plugin options: {exc}") from exc

        parity = ParityOptions(
            input_path=parsed.parity_input_path,
            atol=parsed.parity_atol,
            rtol=parsed.parity_rtol,
        )
        postprocess = PostprocessOptions(
            optimize=parsed.optimize,
            quantize_dynamic=parsed.quantize_dynamic,
            metadata=parsed.metadata,
        )

        model = SklearnModelLoader().load(
            model_path=model_path,
            allow_unsafe=parsed.allow_unsafe,
        )

        converter_options: dict[str, Any] = {"n_features": parsed.n_features}
        if parsed.opset_version is not None:
            converter_options["target_opset"] = parsed.opset_version

        out_path = SklearnModelConverter().convert(
            model=model,
            output_path=output_path,
            options=converter_options,
        )
        SklearnParityChecker().check(model=model, onnx_path=out_path, parity=parity)
        OnnxPostProcessorImpl().run(
            output_path=out_path,
            source_path=model_path,
            framework="sklearn",
            config_metadata={"onnx_converter.n_features": str(parsed.n_features)},
            options=postprocess,
        )
        return out_path
