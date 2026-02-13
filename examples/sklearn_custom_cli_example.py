#!/usr/bin/env python3
"""Train a custom sklearn pipeline, convert via CLI, and validate parity."""

from __future__ import annotations

import subprocess
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from examples.custom_sklearn_transformer import MultiplyByConstant


def main() -> None:
    """Run custom-converter CLI flow with explicit pass/fail checks."""
    X, y = load_iris(return_X_y=True)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    cache_dir = output_dir / "pipeline_cache"
    pipeline = Pipeline(
        [
            ("scale", MultiplyByConstant(factor=1.5)),
            ("clf", LogisticRegression(max_iter=200)),
        ],
        memory=str(cache_dir),
    )
    pipeline.fit(X, y)

    model_path = output_dir / "custom_sklearn.joblib"
    onnx_path = output_dir / "custom_sklearn.onnx"
    joblib.dump(pipeline, model_path)

    command = [
        "convert-to-onnx",
        "sklearn",
        str(model_path),
        str(onnx_path),
        "--n-features",
        str(X.shape[1]),
        "--custom-converter-module",
        "examples/custom_sklearn_transformer.py",
        "--allow-unsafe",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(f"FAIL: CLI conversion failed.\n{result.stderr}")

    if not onnx_path.exists():
        raise SystemExit("FAIL: custom_sklearn.onnx was not created.")
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    batch = X[:16].astype(np.float32)
    sk_pred = np.asarray(pipeline.predict(batch))
    sk_proba = np.asarray(pipeline.predict_proba(batch), dtype=np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": batch})
    onnx_pred = np.asarray(outputs[0])

    if isinstance(outputs[1], list) and outputs[1] and isinstance(outputs[1][0], dict):
        classes = np.asarray(pipeline.classes_)
        onnx_proba = np.array(
            [[row[int(cls)] for cls in classes] for row in outputs[1]], dtype=np.float32
        )
    else:
        onnx_proba = np.asarray(outputs[1], dtype=np.float32)

    if onnx_pred.shape != sk_pred.shape or not np.array_equal(onnx_pred, sk_pred):
        raise SystemExit("FAIL: predicted labels mismatch between sklearn and ONNX.")

    max_abs_diff = float(np.max(np.abs(sk_proba - onnx_proba)))
    print(f"Max |proba diff|: {max_abs_diff:.8f}")
    if not np.allclose(sk_proba, onnx_proba, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: probability mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )

    print(f"PASS: {onnx_path}")


if __name__ == "__main__":
    main()
