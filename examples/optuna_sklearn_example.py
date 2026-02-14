#!/usr/bin/env python3
"""Optuna-tuned sklearn pipeline -> ONNX conversion + parity check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import optuna
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from onnx_converter import convert_sklearn_to_onnx


def _to_prob_matrix(raw_output: object, class_labels: np.ndarray) -> np.ndarray:
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        return np.array(
            [[row[int(cls)] for cls in class_labels] for row in raw_output],
            dtype=np.float32,
        )
    return np.asarray(raw_output, dtype=np.float32)


def _build_pipeline(c_value: float, memory: str | None = None) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ],
        memory=memory,
    )


def main() -> None:
    """Tune a sklearn pipeline with Optuna and validate ONNX parity."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "optuna_logreg.onnx"
    cache_dir = output_dir / "optuna_pipeline_cache"

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    def objective(trial: optuna.Trial) -> float:
        c_value = trial.suggest_float("C", 1e-2, 10.0, log=True)
        pipeline = _build_pipeline(c_value, memory=str(cache_dir))
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        return accuracy_score(y_test, preds)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=12, show_progress_bar=False)

    best_c = float(study.best_params["C"])
    print(f"Best C: {best_c:.6f} (accuracy={study.best_value:.4f})")

    final_model = _build_pipeline(best_c, memory=str(cache_dir))
    final_model.fit(X_train, y_train)

    initial_types = [("input", FloatTensorType([None, X.shape[1]]))]
    convert_sklearn_to_onnx(
        model=final_model,
        output_path=str(onnx_path),
        initial_types=initial_types,
    )

    if not onnx_path.exists():
        raise SystemExit("FAIL: ONNX file was not created.")
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    sklearn_pred = final_model.predict(X_test)
    sklearn_proba = final_model.predict_proba(X_test).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    onnx_outputs = session.run(output_names, {input_name: X_test.astype(np.float32)})

    onnx_pred = np.asarray(onnx_outputs[0])
    onnx_proba = _to_prob_matrix(onnx_outputs[1], final_model.classes_)

    labels_equal = np.array_equal(sklearn_pred, onnx_pred)
    max_abs_diff = float(np.max(np.abs(sklearn_proba - onnx_proba)))

    print("--- Comparison ---")
    print(f"Test samples: {len(X_test)}")
    print(f"Label match: {labels_equal}")
    print(f"Max |proba diff|: {max_abs_diff:.8f}")

    if not labels_equal or max_abs_diff > 1e-5:
        raise SystemExit("Mismatch detected between sklearn and ONNX outputs.")

    print("PASS: Optuna-tuned sklearn pipeline roundtrip verified.")


if __name__ == "__main__":
    main()
