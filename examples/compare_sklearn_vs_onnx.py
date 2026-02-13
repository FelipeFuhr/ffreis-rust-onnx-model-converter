#!/usr/bin/env python3
"""Train a sklearn pipeline, export to ONNX, and compare predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx.common.data_types import FloatTensorType

from examples.custom_sklearn_transformer import MultiplyByConstant
from onnx_converter import convert_sklearn_to_onnx


def _to_proba_matrix(
    raw_output: object,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Normalize ONNX probability output to a 2D float matrix."""
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        return np.array(
            [[row[int(cls)] for cls in class_labels] for row in raw_output],
            dtype=np.float32,
        )
    return np.asarray(raw_output, dtype=np.float32)


def main() -> None:
    """Run a full sklearn-vs-ONNX comparison workflow."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "custom_sklearn.joblib"
    onnx_path = output_dir / "custom_sklearn.onnx"

    # 1) Data + sklearn training
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    cache_dir = output_dir / "compare_pipeline_cache"

    pipeline = Pipeline(
        [
            ("scale", MultiplyByConstant(factor=1.5)),
            ("clf", LogisticRegression(max_iter=200)),
        ],
        memory=str(cache_dir),
    )
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)

    print(f"Saved sklearn model: {model_path}")

    # 2) Export to ONNX
    initial_types = [("input", FloatTensorType([None, X.shape[1]]))]
    convert_sklearn_to_onnx(
        model=pipeline,
        output_path=str(onnx_path),
        initial_types=initial_types,
    )
    print(f"Saved ONNX model: {onnx_path}")

    # 3) Compare sklearn vs ONNX inference
    sklearn_pred = pipeline.predict(X_test)
    sklearn_proba = pipeline.predict_proba(X_test).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    onnx_outputs = session.run(output_names, {input_name: X_test.astype(np.float32)})

    onnx_pred = np.asarray(onnx_outputs[0])
    onnx_proba = _to_proba_matrix(onnx_outputs[1], pipeline.classes_)

    labels_equal = np.array_equal(sklearn_pred, onnx_pred)
    max_abs_diff = float(np.max(np.abs(sklearn_proba - onnx_proba)))

    print("--- Comparison ---")
    print(f"Test samples: {len(X_test)}")
    print(f"Label match: {labels_equal}")
    print(f"Max |proba diff|: {max_abs_diff:.8f}")
    print("Sample sklearn probs[0]:", sklearn_proba[0].tolist())
    print("Sample onnx probs[0]:   ", onnx_proba[0].tolist())

    if not labels_equal or max_abs_diff > 1e-5:
        raise SystemExit("Mismatch detected between sklearn and ONNX outputs.")

    print("Sklearn and ONNX outputs are consistent.")


if __name__ == "__main__":
    main()
