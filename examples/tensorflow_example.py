#!/usr/bin/env python3
"""Example script for converting a tiny TensorFlow/Keras model to ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf

from onnx_converter import convert_tensorflow_to_onnx


def main() -> None:
    """Convert and verify a tiny Keras model."""
    print("=" * 60)
    print("TensorFlow/Keras to ONNX Conversion Example")
    print("=" * 60)

    tf.random.set_seed(7)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,), name="input"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    output_path = Path("outputs/tiny_keras.onnx")
    output_path.parent.mkdir(exist_ok=True)

    convert_tensorflow_to_onnx(
        model=model,
        output_path=str(output_path),
        input_signature=[tf.TensorSpec((None, 4), tf.float32, name="input")],
        opset_version=14,
    )

    if not output_path.exists():
        raise SystemExit("FAIL: ONNX file was not created.")

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph validated.")

    batch = np.array(
        [[0.1, -0.2, 0.3, 0.4], [0.5, -0.6, 0.7, -0.8]],
        dtype=np.float32,
    )

    tf_out = model(batch, training=False).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: batch})[0]

    if tf_out.shape != onnx_out.shape:
        raise SystemExit(
            f"FAIL: shape mismatch (tf={tf_out.shape}, onnx={onnx_out.shape})."
        )

    max_abs_diff = float(np.max(np.abs(tf_out - onnx_out)))
    print(f"Max abs diff: {max_abs_diff:.8f}")
    if not np.allclose(tf_out, onnx_out, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: output mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )

    print(f"PASS: {output_path}")


if __name__ == "__main__":
    main()
