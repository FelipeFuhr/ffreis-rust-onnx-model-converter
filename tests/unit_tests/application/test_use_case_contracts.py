"""Unit tests for application use-case contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter.application.use_cases import (
    build_conversion_options,
    convert_sklearn_file,
    convert_tensorflow_file,
    convert_torch_file,
)
from onnx_converter.errors import ParityError


class _Loader:
    def __init__(self, model: object) -> None:
        self.model = model
        self.calls: list[tuple[Path, bool]] = []

    def load(self, model_path: Path, allow_unsafe: bool = False) -> object:
        self.calls.append((model_path, allow_unsafe))
        return self.model


class _Converter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.calls = 0

    def convert(
        self, model: object, output_path: Path, options: dict[str, object]
    ) -> Path:
        self.calls += 1
        assert model is not None
        assert output_path == self.output_path
        assert isinstance(options, dict)
        return self.output_path


class _Parity:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def check(
        self,
        model: object,
        onnx_path: Path,
        parity: object,
        context: object | None = None,
    ) -> None:
        self.calls += 1
        del model, onnx_path, parity, context
        if self.should_fail:
            raise ParityError("parity failed")


class _Postprocess:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        output_path: Path,
        source_path: Path,
        framework: str,
        config_metadata: dict[str, str],
        options: object,
    ) -> None:
        self.calls += 1
        assert output_path
        assert source_path
        assert framework in {"pytorch", "tensorflow", "sklearn"}
        assert isinstance(config_metadata, dict)
        assert options is not None


def test_torch_use_case_roundtrip_contract(tmp_path: Path) -> None:
    """Verify torch use-case orchestrates loader, converter, parity, and postprocess."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    loader = _Loader(model=object())
    converter = _Converter(output_path=output_path)
    parity = _Parity()
    postprocess = _Postprocess()

    result = convert_torch_file(
        model_path=model_path,
        output_path=output_path,
        input_shape=(1, 4),
        options=build_conversion_options(),
        loader=loader,
        converter=converter,
        parity_checker=parity,
        postprocessor=postprocess,
    )

    assert result.output_path == output_path
    assert converter.calls == 1
    assert parity.calls == 1
    assert postprocess.calls == 1


def test_tensorflow_use_case_parity_failure_is_deterministic(tmp_path: Path) -> None:
    """Verify TensorFlow use-case propagates parity failures deterministically."""
    model_path = tmp_path / "model.h5"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    with pytest.raises(ParityError):
        convert_tensorflow_file(
            model_path=model_path,
            output_path=output_path,
            options=build_conversion_options(),
            loader=_Loader(model=object()),
            converter=_Converter(output_path=output_path),
            parity_checker=_Parity(should_fail=True),
            postprocessor=_Postprocess(),
        )


def test_sklearn_use_case_metadata_path_contract(tmp_path: Path) -> None:
    """Verify sklearn use-case returns expected source path and framework metadata."""
    model_path = tmp_path / "model.skops"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    result = convert_sklearn_file(
        model_path=model_path,
        output_path=output_path,
        n_features=4,
        options=build_conversion_options(),
        loader=_Loader(model=object()),
        converter=_Converter(output_path=output_path),
        parity_checker=_Parity(),
        postprocessor=_Postprocess(),
    )

    assert result.source_path == model_path
    assert result.framework == "sklearn"
