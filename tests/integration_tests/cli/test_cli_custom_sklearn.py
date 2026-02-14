"""CLI integration tests for custom sklearn converters.

Notes
-----
Exercises the custom converter registration hook used by the CLI.

"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module

runner = CliRunner()


def test_cli_sklearn_custom_transformer(tmp_path: Path) -> None:
    """Validate CLI conversion with a dynamically loaded sklearn transformer."""
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    pytest.importorskip("joblib")
    pytest.importorskip("skops")

    import joblib
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    repo_root = Path(__file__).resolve().parents[3]
    converter_module = repo_root / "examples" / "custom_sklearn_transformer.py"
    spec = importlib.util.spec_from_file_location(
        "custom_sklearn_transformer", converter_module
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load custom_sklearn_transformer module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    multiply_by_constant = module.MultiplyByConstant

    features, labels = load_iris(return_X_y=True)
    pipeline = Pipeline(
        [
            ("scale", multiply_by_constant(factor=1.5)),
            ("clf", LogisticRegression(max_iter=200)),
        ],
        memory=None,
    )
    pipeline.fit(features, labels)

    model_path = tmp_path / "custom.joblib"
    output_path = tmp_path / "custom.onnx"
    joblib.dump(pipeline, model_path)

    result = runner.invoke(
        cli_module.app,
        [
            "sklearn",
            str(model_path),
            str(output_path),
            "--n-features",
            str(features.shape[1]),
            "--allow-unsafe",
            "--custom-converter-module",
            str(converter_module),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
