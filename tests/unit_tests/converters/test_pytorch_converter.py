from __future__ import annotations

import sys
import types
from typing import Any
from typing import Callable

import pytest

from onnx_converter import api as api_module
from onnx_converter.errors import ConversionError


class _DummyModel:
    pass


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
    monkeypatch.setattr(onnx_converter.adapters.parity_checkers, "TorchParityChecker", _FakeParityChecker)


def _install_dummy_torch(
    monkeypatch: pytest.MonkeyPatch,
    jit_load: Callable[[str], Any],
    load: Callable[..., Any],
) -> None:
    dummy_torch = types.SimpleNamespace()

    class _Jit:
        @staticmethod
        def load(path: str) -> Any:
            return jit_load(path)

    dummy_torch.jit = _Jit

    def _load(path: str, map_location: Any = None, weights_only: Any = None) -> Any:
        return load(path, map_location=map_location, weights_only=weights_only)

    dummy_torch.load = _load

    monkeypatch.setitem(sys.modules, "torch", dummy_torch)


def test_convert_torch_file_prefers_torchscript(tmp_path, monkeypatch) -> None:
    import onnx_converter
    
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    def jit_load(_path: str) -> _DummyModel:
        return _DummyModel()

    def load(_path: str, map_location: Any = None, weights_only: Any = None) -> _DummyModel:
        raise AssertionError("torch.load should not be called")

    _install_dummy_torch(monkeypatch, jit_load, load)

    # Mock the converter to avoid importing real torch.onnx
    def fake_convert(**kwargs: Any) -> str:
        assert kwargs["input_shape"] == (1, 3)
        return str(output_path)

    monkeypatch.setattr(onnx_converter, "convert_pytorch_to_onnx", fake_convert)
    _mock_converter_dependencies(monkeypatch)

    out = api_module.convert_torch_file_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 3),
        opset_version=14,
        allow_unsafe=False,
    )

    assert out == output_path


def test_convert_torch_file_requires_allow_unsafe(tmp_path, monkeypatch) -> None:
    import onnx_converter
    
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    def jit_load(_path: str) -> _DummyModel:
        raise RuntimeError("fail")

    def load(_path: str, map_location: Any = None, weights_only: Any = None) -> _DummyModel:
        raise RuntimeError("safe load failed")

    _install_dummy_torch(monkeypatch, jit_load, load)

    # Mock the converter to avoid importing real torch.onnx
    def fake_convert(**kwargs: Any) -> str:
        return str(output_path)

    monkeypatch.setattr(onnx_converter, "convert_pytorch_to_onnx", fake_convert)
    _mock_converter_dependencies(monkeypatch)

    with pytest.raises(ConversionError):
        api_module.convert_torch_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_shape=(1, 3),
            opset_version=14,
            allow_unsafe=False,
        )


def test_convert_torch_file_uses_torch_load_when_allowed(tmp_path, monkeypatch) -> None:
    import onnx_converter
    
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    def jit_load(_path: str) -> _DummyModel:
        raise RuntimeError("fail")

    called = {}

    def load(_path: str, map_location: Any = None, weights_only: Any = None) -> _DummyModel:
        called["weights_only"] = weights_only
        return _DummyModel()

    _install_dummy_torch(monkeypatch, jit_load, load)

    # Mock the converter to avoid importing real torch.onnx
    def fake_convert(**kwargs: Any) -> str:
        return str(output_path)

    monkeypatch.setattr(onnx_converter, "convert_pytorch_to_onnx", fake_convert)
    _mock_converter_dependencies(monkeypatch)

    out = api_module.convert_torch_file_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 3),
        opset_version=14,
        allow_unsafe=True,
    )

    assert out == output_path
    assert called["weights_only"] is True


def test_convert_torch_file_falls_back_to_unsafe_only_when_allowed(tmp_path, monkeypatch) -> None:
    import onnx_converter
    
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    def jit_load(_path: str) -> _DummyModel:
        raise RuntimeError("fail")

    called = {"weights_only": []}

    def load(_path: str, map_location: Any = None, weights_only: Any = None) -> _DummyModel:
        called["weights_only"].append(weights_only)
        if weights_only is True:
            raise RuntimeError("safe load failed")
        return _DummyModel()

    _install_dummy_torch(monkeypatch, jit_load, load)

    # Mock the converter to avoid importing real torch.onnx
    def fake_convert(**kwargs: Any) -> str:
        return str(output_path)

    monkeypatch.setattr(onnx_converter, "convert_pytorch_to_onnx", fake_convert)
    _mock_converter_dependencies(monkeypatch)

    out = api_module.convert_torch_file_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 3),
        opset_version=14,
        allow_unsafe=True,
    )

    assert out == output_path
    assert called["weights_only"] == [True, False]
