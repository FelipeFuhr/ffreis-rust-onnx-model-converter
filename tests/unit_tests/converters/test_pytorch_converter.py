from __future__ import annotations

import sys
import types
from typing import Any
from typing import Callable

import pytest

from onnx_converter import api as api_module
from onnx_converter.application.use_cases import build_conversion_options
from onnx_converter.application.use_cases import convert_torch_file
from onnx_converter.errors import ConversionError
from onnx_converter.errors import UnsafeLoadError


class _DummyModel:
    pass


class _MockConverter:
    """Mock converter for testing."""

    def convert(self, model: Any, output_path: Any, options: Any) -> Any:
        return output_path


class _MockParity:
    """Mock parity checker for testing."""

    def check(self, model: Any, onnx_path: Any, parity: Any, context: Any = None) -> None:
        pass


class _MockPostprocess:
    """Mock postprocessor for testing."""

    def run(
        self,
        output_path: Any,
        source_path: Any,
        framework: str,
        config_metadata: Any,
        options: Any,
    ) -> None:
        pass


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
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    jit_load_called = []

    def jit_load(_path: str) -> _DummyModel:
        jit_load_called.append(True)
        return _DummyModel()

    def load(_path: str, map_location: Any = None, weights_only: Any = None) -> _DummyModel:
        raise AssertionError("torch.load should not be called")

    _install_dummy_torch(monkeypatch, jit_load, load)

    class _MockLoaderPrefersTorchScript:
        def load(self, model_path: Any, allow_unsafe: bool = False) -> Any:
            import torch

            return torch.jit.load(str(model_path))

    class _MockConverterCheckOptions(_MockConverter):
        def convert(self, model: Any, output_path: Any, options: Any) -> Any:
            assert isinstance(model, _DummyModel)
            assert options["input_shape"] == (1, 3)
            return output_path

    result = convert_torch_file(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 3),
        options=build_conversion_options(opset_version=14, allow_unsafe=False),
        loader=_MockLoaderPrefersTorchScript(),
        converter=_MockConverterCheckOptions(),
        parity_checker=_MockParity(),
        postprocessor=_MockPostprocess(),
    )

    assert result.output_path == output_path
    assert len(jit_load_called) == 1


def test_convert_torch_file_requires_allow_unsafe(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    def jit_load(_path: str) -> _DummyModel:
        raise RuntimeError("fail")

    def load(_path: str, map_location: Any = None, weights_only: Any = None) -> _DummyModel:
        raise RuntimeError("safe load failed")

    _install_dummy_torch(monkeypatch, jit_load, load)

    class _MockLoaderRequiresUnsafe:
        def load(self, model_path: Any, allow_unsafe: bool = False) -> Any:
            import torch

            # Try jit.load first
            try:
                return torch.jit.load(str(model_path))
            except Exception:
                # Try safe load
                try:
                    return torch.load(str(model_path), map_location="cpu", weights_only=True)
                except Exception as safe_exc:
                    if not allow_unsafe:
                        raise UnsafeLoadError(
                            "TorchScript loading failed and safe torch.load "
                            "fallback was unsuccessful."
                        ) from safe_exc
                    raise

    with pytest.raises((ConversionError, UnsafeLoadError)):
        convert_torch_file(
            model_path=model_path,
            output_path=output_path,
            input_shape=(1, 3),
            options=build_conversion_options(opset_version=14, allow_unsafe=False),
            loader=_MockLoaderRequiresUnsafe(),
            converter=_MockConverter(),
            parity_checker=_MockParity(),
            postprocessor=_MockPostprocess(),
        )


def test_convert_torch_file_uses_torch_load_when_allowed(tmp_path, monkeypatch) -> None:
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

    class _MockLoaderUsesSafeLoad:
        def load(self, model_path: Any, allow_unsafe: bool = False) -> Any:
            import torch

            # Try jit.load first
            try:
                return torch.jit.load(str(model_path))
            except Exception:
                # Try safe load
                return torch.load(str(model_path), map_location="cpu", weights_only=True)

    result = convert_torch_file(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 3),
        options=build_conversion_options(opset_version=14, allow_unsafe=True),
        loader=_MockLoaderUsesSafeLoad(),
        converter=_MockConverter(),
        parity_checker=_MockParity(),
        postprocessor=_MockPostprocess(),
    )

    assert result.output_path == output_path
    assert called["weights_only"] is True


def test_convert_torch_file_falls_back_to_unsafe_only_when_allowed(tmp_path, monkeypatch) -> None:
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

    class _MockLoaderFallsBackToUnsafe:
        def load(self, model_path: Any, allow_unsafe: bool = False) -> Any:
            import torch

            # Try jit.load first
            try:
                return torch.jit.load(str(model_path))
            except Exception:
                # Try safe load
                try:
                    return torch.load(str(model_path), map_location="cpu", weights_only=True)
                except Exception:
                    if allow_unsafe:
                        # Fall back to unsafe load
                        return torch.load(str(model_path), map_location="cpu", weights_only=False)
                    raise

    result = convert_torch_file(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 3),
        options=build_conversion_options(opset_version=14, allow_unsafe=True),
        loader=_MockLoaderFallsBackToUnsafe(),
        converter=_MockConverter(),
        parity_checker=_MockParity(),
        postprocessor=_MockPostprocess(),
    )

    assert result.output_path == output_path
    assert called["weights_only"] == [True, False]
