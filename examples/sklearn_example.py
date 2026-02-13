#!/usr/bin/env python3
"""Examples for converting scikit-learn models/pipelines to ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx.common.data_types import FloatTensorType

from onnx_converter import convert_sklearn_to_onnx


def _to_prob_matrix(raw: Any, classes: np.ndarray) -> np.ndarray:
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return np.array([[row[int(cls)] for cls in classes] for row in raw], dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def _assert_classifier_parity(model: Any, onnx_path: Path, batch: np.ndarray) -> None:
    sklearn_pred = np.asarray(model.predict(batch))
    sklearn_proba = np.asarray(model.predict_proba(batch), dtype=np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": batch.astype(np.float32)})
    onnx_pred = np.asarray(outputs[0])
    onnx_proba = _to_prob_matrix(outputs[1], np.asarray(model.classes_))

    if onnx_pred.shape != sklearn_pred.shape or not np.array_equal(onnx_pred, sklearn_pred):
        raise SystemExit("FAIL: predicted labels mismatch between sklearn and ONNX.")
    if sklearn_proba.shape != onnx_proba.shape:
        raise SystemExit("FAIL: probability tensor shape mismatch.")

    max_abs_diff = float(np.max(np.abs(sklearn_proba - onnx_proba)))
    print(f"Max |proba diff|: {max_abs_diff:.8f}")
    if not np.allclose(sklearn_proba, onnx_proba, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            f"FAIL: probability mismatch (max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )


def example_simple_classifier() -> None:
    """Convert RandomForest and enforce parity checks."""
    print("\n" + "=" * 60)
    print("Example 1: Random Forest Classifier")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    model.fit(X, y)

    output_path = Path("outputs/rf_classifier.onnx")
    output_path.parent.mkdir(exist_ok=True)
    convert_sklearn_to_onnx(
        model=model,
        output_path=str(output_path),
        initial_types=[("input", FloatTensorType([None, 4]))],
    )

    if not output_path.exists():
        raise SystemExit("FAIL: rf_classifier.onnx was not created.")
    onnx.checker.check_model(onnx.load(str(output_path)))

    _assert_classifier_parity(model, output_path, X[:16].astype(np.float32))
    print(f"PASS: {output_path}")


def example_pipeline() -> None:
    """Convert preprocessing+classifier pipeline and enforce parity checks."""
    print("\n" + "=" * 60)
    print("Example 2: Pipeline (Scaler + Random Forest)")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    output_path = Path("outputs/pipeline.onnx")
    output_path.parent.mkdir(exist_ok=True)
    cache_dir = output_path.parent / "pipeline_cache"
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=10,
                    random_state=42,
                    min_samples_leaf=1,
                    max_features="sqrt",
                ),
            ),
        ],
        memory=str(cache_dir),
    )
    pipeline.fit(X, y)

    convert_sklearn_to_onnx(
        model=pipeline,
        output_path=str(output_path),
        initial_types=[("input", FloatTensorType([None, 4]))],
    )

    if not output_path.exists():
        raise SystemExit("FAIL: pipeline.onnx was not created.")
    onnx.checker.check_model(onnx.load(str(output_path)))

    _assert_classifier_parity(pipeline, output_path, X[:16].astype(np.float32))
    print(f"PASS: {output_path}")


def main() -> None:
    """Run all sklearn examples with strict pass/fail criteria."""
    example_simple_classifier()
    example_pipeline()
    print("PASS: sklearn examples complete.")


if __name__ == "__main__":
    main()
