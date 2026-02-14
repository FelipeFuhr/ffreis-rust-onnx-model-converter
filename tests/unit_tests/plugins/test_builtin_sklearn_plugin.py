"""Unit tests for built-in sklearn file plugin behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter.errors import PluginError
from onnx_converter.plugins import builtins
from onnx_converter.plugins.builtins import SklearnFilePlugin


class DummyLoader:
    """Test double for sklearn model loading."""

    def __init__(self) -> None:
        """Initialize call history."""
        self.calls: list[tuple[Path, bool]] = []

    def load(self, model_path: Path, allow_unsafe: bool = False) -> object:
        """Record load call and return placeholder model."""
        self.calls.append((model_path, allow_unsafe))
        return object()


class DummyConverter:
    """Test double for sklearn model conversion."""

    def __init__(self, out: Path) -> None:
        """Initialize output path and call history."""
        self.out = out
        self.calls: list[dict[str, object]] = []

    def convert(
        self,
        model: object,
        output_path: Path,
        options: dict[str, object],
    ) -> Path:
        """Record conversion call and return predetermined output."""
        del model
        self.calls.append({"output_path": output_path, "options": options})
        return self.out


class DummyParity:
    """Test double for parity checking."""

    def __init__(self) -> None:
        """Initialize call history."""
        self.calls: list[object] = []

    def check(self, model: object, onnx_path: Path, parity: object) -> None:
        """Record parity invocation arguments."""
        self.calls.append((model, onnx_path, parity))


class DummyPost:
    """Test double for post-processing pipeline."""

    def __init__(self) -> None:
        """Initialize call history."""
        self.calls: list[object] = []

    def run(
        self,
        output_path: Path,
        source_path: Path,
        framework: str,
        config_metadata: dict[str, str],
        options: object,
    ) -> None:
        """Record post-processing invocation arguments."""
        self.calls.append(
            {
                "output_path": output_path,
                "source_path": source_path,
                "framework": framework,
                "config_metadata": config_metadata,
                "options": options,
            }
        )


def test_requires_n_features() -> None:
    """Require n_features option for sklearn plugin conversion."""
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(Path("model.joblib"), Path("out.onnx"), options={})


def test_rejects_bad_metadata_type() -> None:
    """Reject non-mapping metadata payloads in plugin options."""
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(
            Path("model.joblib"),
            Path("out.onnx"),
            options={"n_features": 4, "metadata": "nope"},
        )


def test_rejects_non_path_parity_input() -> None:
    """Reject parity_input_path values that are not path-like."""
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(
            Path("model.joblib"),
            Path("out.onnx"),
            options={"n_features": 4, "parity_input_path": "bad"},
        )


def test_rejects_bad_opset_type() -> None:
    """Reject non-integer opset values in plugin options."""
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(
            Path("model.joblib"),
            Path("out.onnx"),
            options={"n_features": 4, "opset_version": "14"},
        )


def test_calls_adapters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify plugin wires loader, converter, parity, and postprocess adapters."""
    loader = DummyLoader()
    converter = DummyConverter(out=tmp_path / "out.onnx")
    parity = DummyParity()
    post = DummyPost()

    monkeypatch.setattr(builtins, "SklearnModelLoader", lambda: loader)
    monkeypatch.setattr(builtins, "SklearnModelConverter", lambda: converter)
    monkeypatch.setattr(builtins, "SklearnParityChecker", lambda: parity)
    monkeypatch.setattr(builtins, "OnnxPostProcessorImpl", lambda: post)

    plugin = SklearnFilePlugin()
    out = plugin.convert(
        model_path=tmp_path / "model.skops",
        output_path=tmp_path / "out.onnx",
        options={
            "n_features": 8,
            "allow_unsafe": True,
            "metadata": {"owner": "test"},
            "parity_input_path": tmp_path / "batch.npy",
            "parity_atol": 1e-5,
            "parity_rtol": 1e-4,
            "opset_version": 14,
            "optimize": True,
        },
    )

    assert out == tmp_path / "out.onnx"
    assert loader.calls == [(tmp_path / "model.skops", True)]
    assert converter.calls
    assert parity.calls
    assert post.calls


def test_can_handle_model_type_and_suffix() -> None:
    """Handle explicit model types and known sklearn artifact suffixes."""
    plugin = SklearnFilePlugin()
    assert plugin.can_handle(Path("x.bin"), "sklearn") is True
    assert plugin.can_handle(Path("x.bin"), "autosklearn") is True
    assert plugin.can_handle(Path("x.joblib"), None) is True
    assert plugin.can_handle(Path("x.unknown"), None) is False


def test_calls_adapters_without_opset_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not inject target_opset when plugin opset_version is omitted."""
    loader = DummyLoader()
    converter = DummyConverter(out=tmp_path / "out.onnx")
    parity = DummyParity()
    post = DummyPost()

    monkeypatch.setattr(builtins, "SklearnModelLoader", lambda: loader)
    monkeypatch.setattr(builtins, "SklearnModelConverter", lambda: converter)
    monkeypatch.setattr(builtins, "SklearnParityChecker", lambda: parity)
    monkeypatch.setattr(builtins, "OnnxPostProcessorImpl", lambda: post)

    plugin = SklearnFilePlugin()
    plugin.convert(
        model_path=tmp_path / "model.skops",
        output_path=tmp_path / "out.onnx",
        options={"n_features": 8},
    )

    assert "target_opset" not in converter.calls[0]["options"]
