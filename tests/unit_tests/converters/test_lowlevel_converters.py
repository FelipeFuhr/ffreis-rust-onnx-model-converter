"""Unit tests for low-level converter modules and converter namespace wrappers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from onnx_converter.errors import ConversionError


def _reload_module(module_name: str) -> object:
    """Reload module to pick up monkeypatched dependency modules."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_converters_namespace_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward wrapper calls from onnx_converter.converters.__init__."""
    calls: dict[str, dict[str, object]] = {}

    def fake_pt(**kwargs: object) -> str:
        calls["pt"] = dict(kwargs)
        return "pt.onnx"

    def fake_tf(**kwargs: object) -> str:
        calls["tf"] = dict(kwargs)
        return "tf.onnx"

    def fake_sk(**kwargs: object) -> str:
        calls["sk"] = dict(kwargs)
        return "sk.onnx"

    monkeypatch.setitem(
        sys.modules,
        "onnx_converter.converters.pytorch_converter",
        types.SimpleNamespace(convert_pytorch_to_onnx=fake_pt),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx_converter.converters.tensorflow_converter",
        types.SimpleNamespace(convert_tensorflow_to_onnx=fake_tf),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx_converter.converters.sklearn_converter",
        types.SimpleNamespace(convert_sklearn_to_onnx=fake_sk),
    )

    module = _reload_module("onnx_converter.converters")
    assert module.convert_pytorch_to_onnx(object(), "a.onnx", (1, 3), x=1) == "pt.onnx"
    assert module.convert_tensorflow_to_onnx(object(), "b.onnx", sig="x") == "tf.onnx"
    assert (
        module.convert_sklearn_to_onnx(object(), "c.onnx", target_opset=13) == "sk.onnx"
    )
    assert calls["pt"]["x"] == 1
    assert calls["tf"]["sig"] == "x"
    assert calls["sk"]["target_opset"] == 13


def test_pytorch_converter_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Export PyTorch model using fake torch module and validate output path."""
    export_calls: dict[str, object] = {}

    class FakeModel:
        """Fake model tracking eval() call."""

        eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    def fake_export(*args: object, **kwargs: object) -> None:
        export_calls["args"] = args
        export_calls["kwargs"] = kwargs

    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=object),
        randn=lambda *shape: ("dummy", shape),
        onnx=types.SimpleNamespace(export=fake_export),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.onnx", fake_torch.onnx)

    module = _reload_module("onnx_converter.converters.pytorch_converter")
    model = FakeModel()
    output = tmp_path / "nested" / "model.onnx"

    out = module.convert_pytorch_to_onnx(
        model=model, output_path=str(output), input_shape=(1, 3)
    )

    assert out == str(output)
    assert model.eval_called is True
    assert export_calls["kwargs"]["input_names"] == ["input"]
    assert output.parent.exists()


def test_pytorch_converter_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ConversionError when options fail schema validation."""
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=object),
        randn=lambda *_: None,
        onnx=types.SimpleNamespace(export=lambda **_: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.onnx", fake_torch.onnx)

    module = _reload_module("onnx_converter.converters.pytorch_converter")
    with pytest.raises(ConversionError, match="Invalid PyTorch export options"):
        module.convert_pytorch_to_onnx(
            model=object(),
            output_path="x.onnx",
            input_shape=(),
        )


def test_sklearn_converter_infers_initial_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Infer initial_types from n_features_in_ and write serialized ONNX."""
    calls: dict[str, object] = {}

    class FakeFloatTensorType:
        """Small tensor type placeholder."""

        def __init__(self, shape: list[object]) -> None:
            self.shape = shape

    class FakeOnx:
        """Serialize fake ONNX payload."""

        @staticmethod
        def SerializeToString() -> bytes:
            return b"onnx"

    def fake_convert_sklearn(*args: object, **kwargs: object) -> FakeOnx:
        del args
        calls["kwargs"] = kwargs
        return FakeOnx()

    fake_data_types = types.SimpleNamespace(FloatTensorType=FakeFloatTensorType)
    monkeypatch.setitem(
        sys.modules,
        "skl2onnx",
        types.SimpleNamespace(convert_sklearn=fake_convert_sklearn),
    )
    monkeypatch.setitem(
        sys.modules,
        "skl2onnx.common",
        types.SimpleNamespace(data_types=fake_data_types),
    )
    monkeypatch.setitem(sys.modules, "skl2onnx.common.data_types", fake_data_types)

    module = _reload_module("onnx_converter.converters.sklearn_converter")
    model = types.SimpleNamespace(n_features_in_=4)
    output = tmp_path / "model.onnx"

    out = module.convert_sklearn_to_onnx(model=model, output_path=str(output))

    assert out == str(output)
    assert output.read_bytes() == b"onnx"
    assert calls["kwargs"]["initial_types"][0][0] == "input"


def test_sklearn_converter_requires_initial_types_when_not_inferable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ConversionError when model has no n_features_in_ and no initial_types."""
    fake_data_types = types.SimpleNamespace(FloatTensorType=lambda *_: object())
    monkeypatch.setitem(
        sys.modules,
        "skl2onnx",
        types.SimpleNamespace(convert_sklearn=lambda **_: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "skl2onnx.common",
        types.SimpleNamespace(data_types=fake_data_types),
    )
    monkeypatch.setitem(sys.modules, "skl2onnx.common.data_types", fake_data_types)

    module = _reload_module("onnx_converter.converters.sklearn_converter")
    with pytest.raises(ConversionError, match="Could not infer input types"):
        module.convert_sklearn_to_onnx(model=object(), output_path="x.onnx")


def test_tensorflow_converter_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover SavedModel and Keras branches in TensorFlow converter."""
    calls: dict[str, object] = {}

    class FakeModelBase:
        """Minimal fake tf.keras.Model."""

        input_shape = (None, 4)
        outputs = [types.SimpleNamespace(name="out:0")]

    def tensor_spec(
        shape: object, dtype: object, name: str
    ) -> tuple[object, object, str]:
        return (shape, dtype, name)

    fake_tf = types.SimpleNamespace(
        float32="float32",
        TensorSpec=tensor_spec,
        keras=types.SimpleNamespace(Model=FakeModelBase),
    )
    fake_tf2onnx = types.SimpleNamespace(
        convert=types.SimpleNamespace(
            from_saved_model=lambda *args, **kwargs: calls.setdefault(
                "saved", (args, kwargs)
            ),
            from_keras=lambda *args, **kwargs: calls.setdefault(
                "keras", (args, kwargs)
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "tf2onnx", fake_tf2onnx)

    module = _reload_module("onnx_converter.converters.tensorflow_converter")
    assert (
        module.convert_tensorflow_to_onnx(model="saved", output_path="a.onnx")
        == "a.onnx"
    )
    model = FakeModelBase()
    assert (
        module.convert_tensorflow_to_onnx(model=model, output_path="b.onnx") == "b.onnx"
    )
    assert "saved" in calls
    assert "keras" in calls
    assert hasattr(model, "output_names")

    with pytest.raises(ValueError, match="Unsupported model type"):
        module.convert_tensorflow_to_onnx(model=object(), output_path="x.onnx")
