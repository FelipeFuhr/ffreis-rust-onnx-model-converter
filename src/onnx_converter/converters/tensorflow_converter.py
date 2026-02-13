"""TensorFlow/Keras-to-ONNX conversion utilities."""

from __future__ import annotations

import os
from typing import Any

import tensorflow as tf
import tf2onnx


def _ensure_output_dir(output_path: str) -> None:
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )


def _ensure_keras_output_names(model: tf.keras.Model) -> None:
    """Set output_names for Keras models when absent (Keras 3 compatibility)."""
    if not hasattr(model, "output_names") and hasattr(model, "outputs"):
        model.output_names = [tensor.name.split(":")[0] for tensor in model.outputs]


def _build_default_signature(model: tf.keras.Model) -> list[tf.TensorSpec] | None:
    """Infer tf2onnx input signature from model input_shape."""
    if not hasattr(model, "input_shape"):
        return None

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        return [
            tf.TensorSpec(shape, tf.float32, name=f"input_{i}")
            for i, shape in enumerate(input_shape)
        ]
    return [tf.TensorSpec(input_shape, tf.float32, name="input")]


def _convert_saved_model(
    model_path: str,
    output_path: str,
    input_signature: list[tf.TensorSpec] | None,
    opset_version: int,
    **kwargs: Any,
) -> None:
    tf2onnx.convert.from_saved_model(
        model_path,
        input_signature=input_signature,
        opset=opset_version,
        output_path=output_path,
        **kwargs,
    )


def _convert_keras_model(
    model: tf.keras.Model,
    output_path: str,
    input_signature: list[tf.TensorSpec] | None,
    opset_version: int,
    **kwargs: Any,
) -> None:
    _ensure_keras_output_names(model)
    resolved_signature = input_signature or _build_default_signature(model)

    tf2onnx.convert.from_keras(
        model,
        input_signature=resolved_signature,
        opset=opset_version,
        output_path=output_path,
        **kwargs,
    )


def convert_tensorflow_to_onnx(
    model: str | tf.keras.Model,
    output_path: str,
    input_signature: list[tf.TensorSpec] | None = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    """Convert a TensorFlow or Keras model to ONNX format.

    Parameters
    ----------
    model
        TensorFlow/Keras model instance or SavedModel path.
    output_path : str
        Path where the ONNX model will be written.
    input_signature : list[tf.TensorSpec], optional
        Input tensor signature passed to ``tf2onnx``.
    opset_version : int, default=14
        ONNX opset version used by ``tf2onnx``.
    **kwargs
        Additional keyword arguments forwarded to ``tf2onnx.convert``.

    Returns
    -------
    str
        Path to the saved ONNX model.

    Raises
    ------
    ValueError
        If ``model`` is neither a SavedModel path nor a ``tf.keras.Model``.
    """
    _ensure_output_dir(output_path)

    if isinstance(model, str):
        _convert_saved_model(
            model_path=model,
            output_path=output_path,
            input_signature=input_signature,
            opset_version=opset_version,
            **kwargs,
        )
        return output_path

    if not isinstance(model, tf.keras.Model):
        raise ValueError(f"Unsupported model type: {type(model)}")

    _convert_keras_model(
        model=model,
        output_path=output_path,
        input_signature=input_signature,
        opset_version=opset_version,
        **kwargs,
    )

    return output_path
