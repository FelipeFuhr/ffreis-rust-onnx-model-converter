"""
TensorFlow/Keras to ONNX Converter
"""
import os
from typing import List, Optional

import tensorflow as tf
import tf2onnx


def convert_tensorflow_to_onnx(
    model,
    output_path: str,
    input_signature: Optional[List[tf.TensorSpec]] = None,
    opset_version: int = 14,
    **kwargs
) -> str:
    """
    Convert a TensorFlow or Keras model to ONNX format.

    Args:
        model: TensorFlow/Keras model to convert (can be tf.keras.Model or path to SavedModel)
        output_path: Path where the ONNX model will be saved
        input_signature: List of TensorSpec defining input shapes and dtypes
        opset_version: ONNX opset version (default: 14)
        **kwargs: Additional arguments to pass to tf2onnx.convert

    Returns:
        Path to the saved ONNX model

    Example:
        >>> model = tf.keras.applications.MobileNetV2(weights='imagenet')
        >>> input_spec = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")]
        >>> convert_tensorflow_to_onnx(model, "mobilenet.onnx", input_signature=input_spec)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if isinstance(model, str):
        tf2onnx.convert.from_saved_model(
            model,
            input_signature=input_signature,
            opset=opset_version,
            output_path=output_path,
            **kwargs
        )
    elif isinstance(model, tf.keras.Model):
        if input_signature is None:
            if hasattr(model, "input_shape"):
                input_shape = model.input_shape
                if isinstance(input_shape, list):
                    input_signature = [
                        tf.TensorSpec(shape, tf.float32, name=f"input_{i}")
                        for i, shape in enumerate(input_shape)
                    ]
                else:
                    input_signature = [tf.TensorSpec(input_shape, tf.float32, name="input")]

        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset_version,
            output_path=output_path,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    print(f"✓ TensorFlow/Keras model successfully converted to ONNX: {output_path}")
    return output_path
