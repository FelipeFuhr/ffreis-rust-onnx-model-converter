"""Unit tests for converter daemon core helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter.converter import core
from onnx_converter.converter.core import ConversionRequest, digest_bytes


def test_convert_artifact_bytes_returns_integrity_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify conversion returns stable input/output hashes."""

    def fake_run_conversion(
        input_path: Path,
        request: ConversionRequest,
        output_path: Path,
    ) -> Path:
        _ = (input_path, request)
        output_path.write_bytes(b"onnx-output")
        return output_path

    monkeypatch.setattr(core, "run_conversion", fake_run_conversion)
    payload = b"model-bytes"
    request = ConversionRequest(
        framework="sklearn",
        filename="model.joblib",
        expected_sha256=digest_bytes(payload),
        n_features=4,
    )
    input_sha, outcome = core.convert_artifact_bytes(payload, request)
    assert input_sha == digest_bytes(payload)
    assert outcome.output_sha256 == digest_bytes(b"onnx-output")
    assert outcome.output_bytes == b"onnx-output"
    assert outcome.output_size_bytes == len(b"onnx-output")


def test_convert_artifact_bytes_rejects_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when provided input digest does not match payload."""

    def fake_run_conversion(
        input_path: Path,
        request: ConversionRequest,
        output_path: Path,
    ) -> Path:
        _ = (input_path, request)
        output_path.write_bytes(b"x")
        return output_path

    monkeypatch.setattr(core, "run_conversion", fake_run_conversion)
    request = ConversionRequest(
        framework="tensorflow",
        filename="saved_model.zip",
        expected_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        core.convert_artifact_bytes(b"actual", request)


def test_run_conversion_validates_framework_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate framework-specific requirements before conversion."""
    dummy_path = Path("a.bin")
    dummy_output = Path("b.onnx")

    with pytest.raises(ValueError, match="input_shape is required"):
        core.run_conversion(
            input_path=dummy_path,
            request=ConversionRequest(framework="pytorch", filename="m.pt"),
            output_path=dummy_output,
        )

    with pytest.raises(ValueError, match="n_features must be > 0"):
        core.run_conversion(
            input_path=dummy_path,
            request=ConversionRequest(framework="sklearn", filename="m.joblib"),
            output_path=dummy_output,
        )
