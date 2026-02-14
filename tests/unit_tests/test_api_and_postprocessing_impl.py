"""Unit tests for file-based API wrappers and infrastructure post-processor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import onnx_converter.api as api_module
import onnx_converter.infrastructure.postprocessing as post_module
from onnx_converter.application.options import PostprocessOptions


def test_api_file_wrappers_forward_and_return_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward API wrapper calls into use-case layer and return result.output_path."""
    recorded: dict[str, object] = {}

    def fake_build_options(**kwargs: object) -> object:
        recorded.setdefault("options_calls", []).append(kwargs)
        return {"opts": kwargs}

    def fake_convert_torch_file(**kwargs: object) -> object:
        recorded["torch"] = kwargs
        return SimpleNamespace(output_path=tmp_path / "torch.onnx")

    def fake_convert_tf_file(**kwargs: object) -> object:
        recorded["tf"] = kwargs
        return SimpleNamespace(output_path=tmp_path / "tf.onnx")

    def fake_convert_sk_file(**kwargs: object) -> object:
        recorded["sk"] = kwargs
        return SimpleNamespace(output_path=tmp_path / "sk.onnx")

    def fake_convert_custom_file(**kwargs: object) -> object:
        recorded["custom"] = kwargs
        return SimpleNamespace(output_path=tmp_path / "custom.onnx")

    monkeypatch.setattr(api_module, "build_conversion_options", fake_build_options)
    monkeypatch.setattr(api_module, "convert_torch_file", fake_convert_torch_file)
    monkeypatch.setattr(api_module, "convert_tensorflow_file", fake_convert_tf_file)
    monkeypatch.setattr(api_module, "convert_sklearn_file", fake_convert_sk_file)
    monkeypatch.setattr(api_module, "convert_custom_file", fake_convert_custom_file)

    assert (
        api_module.convert_torch_file_to_onnx(
            model_path=tmp_path / "a.pt",
            output_path=tmp_path / "a.onnx",
            input_shape=(1, 3),
            parity_atol=1e-6,
        )
        == tmp_path / "torch.onnx"
    )
    assert (
        api_module.convert_tf_path_to_onnx(
            model_path=tmp_path / "m.h5",
            output_path=tmp_path / "b.onnx",
            parity_rtol=1e-3,
        )
        == tmp_path / "tf.onnx"
    )
    assert (
        api_module.convert_sklearn_file_to_onnx(
            model_path=tmp_path / "m.joblib",
            output_path=tmp_path / "c.onnx",
            n_features=8,
        )
        == tmp_path / "sk.onnx"
    )
    assert (
        api_module.convert_custom_file_to_onnx(
            model_path=tmp_path / "m.bin",
            output_path=tmp_path / "d.onnx",
            options=None,
        )
        == tmp_path / "custom.onnx"
    )

    assert recorded["torch"]["input_shape"] == (1, 3)  # type: ignore[index]
    assert recorded["sk"]["n_features"] == 8  # type: ignore[index]
    assert recorded["custom"]["options"] == {}  # type: ignore[index]


def test_postprocessor_impl_runs_optional_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Apply standard metadata and optional metadata/optimize/quantize steps."""
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        post_module,
        "add_standard_metadata",
        lambda **kwargs: calls.append(("standard", kwargs)),
    )
    monkeypatch.setattr(
        post_module,
        "add_onnx_metadata",
        lambda *args: calls.append(("meta", args)),
    )
    monkeypatch.setattr(
        post_module,
        "optimize_onnx_graph",
        lambda path: calls.append(("opt", path)),
    )
    monkeypatch.setattr(
        post_module,
        "quantize_onnx_dynamic",
        lambda path: calls.append(("quant", path)),
    )

    post_module.OnnxPostProcessorImpl().run(
        output_path=tmp_path / "m.onnx",
        source_path=tmp_path / "src.bin",
        framework="sklearn",
        config_metadata={"k": "v"},
        options=PostprocessOptions(
            optimize=True,
            quantize_dynamic=True,
            metadata={"x": "1"},
        ),
    )

    kinds = [kind for kind, _payload in calls]
    assert kinds == ["standard", "meta", "opt", "quant"]
