"""Integration tests for the sklearn converter backend."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sklearn_convert(tmp_path: Path) -> None:
    """Convert a fitted sklearn estimator and verify ONNX output exists."""
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")

    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from onnx_converter.converters.sklearn_converter import convert_sklearn_to_onnx

    features, labels = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(features, labels)

    output_path = tmp_path / "model.onnx"
    initial_types = [("input", FloatTensorType([None, features.shape[1]]))]

    out = convert_sklearn_to_onnx(
        model=model,
        output_path=str(output_path),
        initial_types=initial_types,
    )

    assert output_path.exists()
    assert str(output_path) == out
