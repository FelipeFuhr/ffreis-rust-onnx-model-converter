"""End-to-end tests for the public converter CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import onnx
import pytest


def test_cli_sklearn_roundtrip(tmp_path: Path) -> None:
    """Run a full sklearn -> ONNX conversion through the public CLI."""
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    features, labels = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200, random_state=42).fit(features, labels)

    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    joblib.dump(model, model_path)

    cmd = [
        "convert-to-onnx",
        "sklearn",
        str(model_path),
        str(onnx_path),
        "--n-features",
        str(features.shape[1]),
        "--allow-unsafe",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert onnx_path.exists()
    onnx.checker.check_model(onnx.load(str(onnx_path)))
