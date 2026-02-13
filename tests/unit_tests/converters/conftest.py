"""Shared test helpers for converter tests."""

from __future__ import annotations

import pytest

import onnx_converter.adapters.parity_checkers
import onnx_converter.postprocess


class FakeParityChecker:
    """Fake parity checker for testing."""

    def check(self, *args, **kwargs):
        # Intentionally no-op for converter dependency isolation in unit tests.
        del args, kwargs


def mock_converter_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    framework: str = "torch",
) -> None:
    """Mock converter, postprocessor, and parity checker dependencies.

    Args:
        monkeypatch: pytest monkeypatch fixture
        framework: Framework name ("torch" or "tensorflow")
    """

    # Mock postprocess functions to avoid loading ONNX files
    monkeypatch.setattr(
        onnx_converter.postprocess,
        "add_standard_metadata",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        onnx_converter.postprocess,
        "add_onnx_metadata",
        lambda *args, **kwargs: None,
    )

    # Mock parity checker to avoid dependencies
    if framework == "torch":
        monkeypatch.setattr(
            onnx_converter.adapters.parity_checkers,
            "TorchParityChecker",
            FakeParityChecker,
        )
    elif framework == "tensorflow":
        monkeypatch.setattr(
            onnx_converter.adapters.parity_checkers,
            "TensorflowParityChecker",
            FakeParityChecker,
        )
