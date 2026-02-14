#!/usr/bin/env python3
"""Train autosklearn, convert to ONNX, and validate prediction parity."""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def _select_automl_class(flavor: str):
    if flavor == "2":
        try:
            from autosklearn.experimental.askl2 import AutoSklearn2Classifier

            return AutoSklearn2Classifier
        except Exception as exc:
            raise SystemExit(
                "FAIL: autosklearn flavor=2 requested but "
                "AutoSklearn2Classifier is unavailable."
            ) from exc

    from autosklearn.classification import AutoSklearnClassifier

    return AutoSklearnClassifier


def _filtered_kwargs(cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _to_prob_matrix(raw: object, classes: np.ndarray) -> np.ndarray:
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return np.array(
            [[row[int(cls)] for cls in classes] for row in raw], dtype=np.float32
        )
    return np.asarray(raw, dtype=np.float32)


def main() -> None:
    """Train AutoSklearn, export ONNX, and validate prediction parity."""
    flavor = os.environ.get("AUTOSKLEARN_FLAVOR", "1").strip()
    if flavor not in {"1", "2"}:
        raise SystemExit("FAIL: AUTOSKLEARN_FLAVOR must be '1' or '2'.")

    out_dir = Path("outputs/autosklearn")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"autosklearn_v{flavor}.joblib"
    onnx_path = out_dir / f"autosklearn_v{flavor}.onnx"

    X, y = make_classification(
        n_samples=450,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        weights=[0.65, 0.25, 0.10],
        random_state=42,
    )
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    automl_cls = _select_automl_class(flavor)
    base_kwargs: dict[str, object] = {
        "time_left_for_this_task": 60,
        "per_run_time_limit": 20,
        "seed": 42,
        "n_jobs": 1,
        "initial_configurations_via_metalearning": 0,
        "ensemble_kwargs": {"ensemble_size": 1},
        "include": {
            "classifier": ["random_forest"],
            "feature_preprocessor": ["no_preprocessing"],
        },
    }
    automl = automl_cls(**_filtered_kwargs(automl_cls, base_kwargs))
    automl.fit(X_train, y_train)

    joblib.dump(automl, model_path)
    reloaded = joblib.load(model_path)

    convert_cmd = [
        "convert-to-onnx",
        "custom",
        str(model_path),
        str(onnx_path),
        "--model-type",
        "autosklearn",
        "--plugin-module",
        "examples/autosklearn_plugin.py",
        "--plugin-name",
        "autosklearn",
        "--n-features",
        str(X.shape[1]),
        "--allow-unsafe",
    ]
    result = subprocess.run(convert_cmd, check=False, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(f"FAIL: conversion command failed.\n{result.stderr}")

    if not onnx_path.exists():
        raise SystemExit("FAIL: ONNX artifact was not generated.")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    sk_pred = np.asarray(reloaded.predict(X_test))
    sk_proba = np.asarray(reloaded.predict_proba(X_test), dtype=np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_test.astype(np.float32)})
    onnx_pred = np.asarray(outputs[0])
    onnx_proba = _to_prob_matrix(outputs[1], np.asarray(reloaded.classes_))

    if onnx_pred.shape != sk_pred.shape or not np.array_equal(onnx_pred, sk_pred):
        raise SystemExit(
            "FAIL: label predictions mismatch between autosklearn and ONNX."
        )

    if onnx_proba.shape != sk_proba.shape:
        raise SystemExit(
            "FAIL: probability shape mismatch "
            f"({onnx_proba.shape} vs {sk_proba.shape})."
        )

    max_abs_diff = float(np.max(np.abs(sk_proba - onnx_proba)))
    print(f"Max |proba diff|: {max_abs_diff:.8f}")
    if not np.allclose(sk_proba, onnx_proba, atol=1e-4, rtol=1e-3):
        raise SystemExit(
            "FAIL: probability mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-4, rtol=1e-3)."
        )

    accuracy = float((sk_pred == y_test).mean())
    print(f"Reference artifact accuracy on holdout: {accuracy:.4f}")
    print(f"PASS: autosklearn flavor {flavor} roundtrip validated.")


if __name__ == "__main__":
    main()
