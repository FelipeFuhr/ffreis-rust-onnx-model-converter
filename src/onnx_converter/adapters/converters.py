"""Model-to-ONNX converters implementing application ports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from onnx_converter.errors import UnsupportedModelError


class TorchModelConverter:
    """Convert torch model to ONNX."""

    def convert(
        self,
        model: Any,
        output_path: Path,
        options: Mapping[str, Any],
    ) -> Path:
        """Convert a PyTorch model to ONNX.

        Parameters
        ----------
        model : Any
            In-memory PyTorch model object.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, Any]
            Conversion options passed to the backend converter.

        Returns
        -------
        Path
            Path to the generated ONNX artifact.
        """
        from onnx_converter import convert_pytorch_to_onnx

        out = convert_pytorch_to_onnx(
            model=model,
            output_path=str(output_path),
            input_shape=tuple(options["input_shape"]),
            input_names=options.get("input_names"),
            output_names=options.get("output_names"),
            dynamic_axes=options.get("dynamic_axes"),
            opset_version=int(options.get("opset_version", 14)),
        )
        return Path(out)


class TensorflowModelConverter:
    """Convert TensorFlow/Keras model to ONNX."""

    def convert(
        self,
        model: Any,
        output_path: Path,
        options: Mapping[str, Any],
    ) -> Path:
        """Convert a TensorFlow model to ONNX.

        Parameters
        ----------
        model : Any
            TensorFlow/Keras model object.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, Any]
            Conversion options passed to the backend converter.

        Returns
        -------
        Path
            Path to the generated ONNX artifact.
        """
        from onnx_converter import convert_tensorflow_to_onnx

        out = convert_tensorflow_to_onnx(
            model=model,
            output_path=str(output_path),
            input_signature=options.get("input_signature"),
            opset_version=int(options.get("opset_version", 14)),
        )
        return Path(out)


class SklearnModelConverter:
    """Convert sklearn model to ONNX."""

    def convert(
        self,
        model: Any,
        output_path: Path,
        options: Mapping[str, Any],
    ) -> Path:
        """Convert a scikit-learn model to ONNX.

        Parameters
        ----------
        model : Any
            Trained scikit-learn estimator or pipeline.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, Any]
            Conversion options, including ``n_features``.

        Returns
        -------
        Path
            Path to the generated ONNX artifact.
        """
        n_features = options.get("n_features")
        if not isinstance(n_features, int) or n_features <= 0:
            raise UnsupportedModelError(
                "n_features is required for sklearn conversion."
            )
        initial_types = options.get("initial_types")
        if initial_types is None:
            try:
                from skl2onnx.common.data_types import FloatTensorType
            except Exception as exc:
                raise UnsupportedModelError(
                    "skl2onnx is required for sklearn conversion."
                ) from exc
            initial_types = [("input", FloatTensorType([None, n_features]))]

        from onnx_converter import convert_sklearn_to_onnx

        out = convert_sklearn_to_onnx(
            model=model,
            output_path=str(output_path),
            initial_types=initial_types,
            target_opset=options.get("target_opset"),
        )
        return Path(out)
