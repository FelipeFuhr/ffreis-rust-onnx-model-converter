"""Scikit-learn-to-ONNX conversion utilities."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from onnx_converter.errors import ConversionError
from onnx_converter.schemas import SklearnConversionConfig
from onnx_converter.types import ModelArtifact, OptionValue, SklearnInitialTypeLike


def convert_sklearn_to_onnx(
    model: ModelArtifact,
    output_path: str,
    initial_types: list[tuple[str, SklearnInitialTypeLike]] | None = None,
    target_opset: int | None = None,
    **kwargs: OptionValue,
) -> str:
    """Convert a scikit-learn model or pipeline to ONNX format.

    Parameters
    ----------
    model : ModelArtifact
        Scikit-learn model or pipeline instance to convert.
    output_path : str
        Path where the ONNX model will be written.
    initial_types : list[tuple[str, SklearnInitialTypeLike]], optional
        Input type declarations expected by ``skl2onnx``.
        When omitted, input types are inferred from ``model.n_features_in_``.
    target_opset : int, optional
        ONNX opset version used by ``skl2onnx``.
    **kwargs
        Additional keyword arguments forwarded to ``convert_sklearn``.

    Returns
    -------
    str
        Path to the saved ONNX model.

    Raises
    ------
    ConversionError
        If input types cannot be inferred and ``initial_types`` is not provided.
        Also raised when converter configuration types are invalid.
    """
    try:
        config = SklearnConversionConfig(
            output_path=Path(output_path),
            initial_types=initial_types,
            target_opset=target_opset,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid sklearn export options: {exc}") from exc

    output_path_str = str(config.output_path)
    os.makedirs(
        os.path.dirname(output_path_str) if os.path.dirname(output_path_str) else ".",
        exist_ok=True,
    )

    resolved_initial_types = config.initial_types
    if resolved_initial_types is None:
        if hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            resolved_initial_types = [("input", FloatTensorType([None, n_features]))]
        else:
            raise ConversionError(
                "Could not infer input types. "
                "Please provide 'initial_types' parameter.\n"
                "Example: [('input', FloatTensorType([None, n_features]))]"
            )

    onx = convert_sklearn(
        model,
        initial_types=resolved_initial_types,
        target_opset=config.target_opset,
        **kwargs,
    )

    with open(output_path_str, "wb") as handle:
        handle.write(onx.SerializeToString())

    return output_path_str
