"""CLI integration tests for custom sklearn converters.

Notes
-----
Exercises the custom converter registration hook used by the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module


runner = CliRunner()


def test_cli_sklearn_custom_transformer(tmp_path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    pytest.importorskip("joblib")
    pytest.importorskip("skops")

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import joblib

    from examples.custom_sklearn_transformer import MultiplyByConstant

    X, y = load_iris(return_X_y=True)
    cache_dir = tmp_path / "pipeline_cache"
    pipeline = Pipeline(
        [
            ("scale", MultiplyByConstant(factor=1.5)),
            ("clf", LogisticRegression(max_iter=200)),
        ],
        memory=str(cache_dir),
    )
    pipeline.fit(X, y)

    model_path = tmp_path / "custom.joblib"
    output_path = tmp_path / "custom.onnx"
    joblib.dump(pipeline, model_path)

    repo_root = Path(__file__).resolve().parents[3]
    converter_module = repo_root / "examples" / "custom_sklearn_transformer.py"

    result = runner.invoke(
        cli_module.app,
        [
            "sklearn",
            str(model_path),
            str(output_path),
            "--n-features",
            str(X.shape[1]),
            "--custom-converter-module",
            str(converter_module),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
