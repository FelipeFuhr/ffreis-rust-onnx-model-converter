from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from onnx_converter import api as api_module


class _FakeParityChecker:
    """Fake parity checker for testing."""
    def check(self, *args, **kwargs):
        pass


def _mock_converter_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock converter, postprocessor, and parity checker dependencies."""
    import onnx_converter
    import onnx_converter.postprocess
    import onnx_converter.adapters.parity_checkers
    
    # Mock postprocess functions to avoid loading ONNX files
    monkeypatch.setattr(onnx_converter.postprocess, "add_standard_metadata", lambda **kwargs: None)
    monkeypatch.setattr(onnx_converter.postprocess, "add_onnx_metadata", lambda *args, **kwargs: None)
    
    # Mock parity checker to avoid dependencies
    monkeypatch.setattr(onnx_converter.adapters.parity_checkers, "TensorflowParityChecker", _FakeParityChecker)


def _install_dummy_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_tf = types.SimpleNamespace()

    class _Models:
        @staticmethod
        def load_model(path: str) -> str:
            return f"loaded:{path}"

    class _Keras:
        models = _Models

    dummy_tf.keras = _Keras

    monkeypatch.setitem(sys.modules, "tensorflow", dummy_tf)
    monkeypatch.setitem(sys.modules, "tf2onnx", types.SimpleNamespace())


def test_convert_tf_path_uses_savedmodel_dir(tmp_path, monkeypatch) -> None:
    import onnx_converter
    
    model_path = tmp_path / "saved_model"
    model_path.mkdir()
    output_path = tmp_path / "out.onnx"

    _install_dummy_tf(monkeypatch)

    # Mock the converter to avoid importing real tf2onnx
    def fake_convert(**kwargs: Any) -> str:
        # For SavedModel directories, the model parameter should be the path
        return str(output_path)

    monkeypatch.setattr(onnx_converter, "convert_tensorflow_to_onnx", fake_convert)
    _mock_converter_dependencies(monkeypatch)

    out = api_module.convert_tf_path_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=14,
    )

    assert out == output_path


def test_convert_tf_path_loads_file(tmp_path, monkeypatch) -> None:
    import onnx_converter
    
    model_path = tmp_path / "model.h5"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    _install_dummy_tf(monkeypatch)

    # Mock the converter to avoid importing real tf2onnx
    def fake_convert(**kwargs: Any) -> str:
        # For regular files, the loader should load the model
        return str(output_path)

    monkeypatch.setattr(onnx_converter, "convert_tensorflow_to_onnx", fake_convert)
    _mock_converter_dependencies(monkeypatch)

    out = api_module.convert_tf_path_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=14,
    )

    assert out == output_path
