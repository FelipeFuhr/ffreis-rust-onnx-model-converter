"""Model-to-ONNX converters implementing application ports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from onnx_converter.errors import UnsupportedModelError


class TorchModelConverter:
    """Convert torch model to ONNX."""

    def convert(
        self,
        model: object,
        output_path: Path,
        options: Mapping[str, object],
    ) -> Path:
        """Convert a PyTorch model to ONNX.

        Parameters
        ----------
        model : Any
            In-memory PyTorch model object.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, object]
            Conversion options passed to the backend converter.

        Returns
        -------
        Path
            Path to the generated ONNX artifact.
        """
        from onnx_converter import convert_pytorch_to_onnx

        raw_input_shape = options.get("input_shape")
        if not isinstance(raw_input_shape, (list, tuple)):
            raise UnsupportedModelError("input_shape is required for torch conversion.")
        input_shape = tuple(int(value) for value in raw_input_shape)

        raw_input_names = options.get("input_names")
        input_names = (
            cast(list[str], raw_input_names)
            if isinstance(raw_input_names, list)
            else None
        )

        raw_output_names = options.get("output_names")
        output_names = (
            cast(list[str], raw_output_names)
            if isinstance(raw_output_names, list)
            else None
        )

        raw_dynamic_axes = options.get("dynamic_axes")
        dynamic_axes = (
            cast(dict[str, dict[int, str]], raw_dynamic_axes)
            if isinstance(raw_dynamic_axes, dict)
            else None
        )

        raw_opset = options.get("opset_version", 14)
        if not isinstance(raw_opset, int):
            raise UnsupportedModelError("opset_version must be an integer.")

        out = convert_pytorch_to_onnx(
            model=model,
            output_path=str(output_path),
            input_shape=input_shape,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=raw_opset,
        )
        return Path(out)


class TensorflowModelConverter:
    """Convert TensorFlow/Keras model to ONNX."""

    def convert(
        self,
        model: object,
        output_path: Path,
        options: Mapping[str, object],
    ) -> Path:
        """Convert a TensorFlow model to ONNX.

        Parameters
        ----------
        model : Any
            TensorFlow/Keras model object.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, object]
            Conversion options passed to the backend converter.

        Returns
        -------
        Path
            Path to the generated ONNX artifact.
        """
        from onnx_converter import convert_tensorflow_to_onnx

        raw_opset = options.get("opset_version", 14)
        if not isinstance(raw_opset, int):
            raise UnsupportedModelError("opset_version must be an integer.")
        raw_signature = options.get("input_signature")
        input_signature = (
            cast(list[object], raw_signature)
            if isinstance(raw_signature, list)
            else None
        )

        out = convert_tensorflow_to_onnx(
            model=model,
            output_path=str(output_path),
            input_signature=input_signature,
            opset_version=raw_opset,
        )
        return Path(out)


class SklearnModelConverter:
    """Convert sklearn model to ONNX."""

    def convert(
        self,
        model: object,
        output_path: Path,
        options: Mapping[str, object],
    ) -> Path:
        """Convert a scikit-learn model to ONNX.

        Parameters
        ----------
        model : Any
            Trained scikit-learn estimator or pipeline.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, object]
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
            target_opset=(
                cast(int, options["target_opset"])
                if isinstance(options.get("target_opset"), int)
                else None
            ),
        )
        return Path(out)
