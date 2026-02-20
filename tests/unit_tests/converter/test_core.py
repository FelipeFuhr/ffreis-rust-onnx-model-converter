"""Unit tests for converter daemon core helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter.converter import core
from onnx_converter.converter.core import ConversionRequest, digest_bytes
from onnx_converter.errors import ConversionError


def test_digest_file_and_write_bytes_roundtrip(tmp_path: Path) -> None:
    """Compute the same digest from bytes and the written file."""
    payload = b"abc123"
    output = tmp_path / "nested" / "file.bin"
    output_sha = core.write_bytes_to_file(output, payload)
    assert output_sha == digest_bytes(payload)
    assert core.digest_file(output) == digest_bytes(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("A" * 64, "a" * 64),
    ],
)
def test_normalize_sha256(value: str | None, expected: str | None) -> None:
    """Normalize expected SHA values and treat blanks as absent."""
    assert core.normalize_sha256(value) == expected


def test_normalize_sha256_rejects_invalid_value() -> None:
    """Reject non-hex and wrong-length SHA values."""
    with pytest.raises(ValueError, match="64-character hex digest"):
        core.normalize_sha256("xyz")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("model.joblib", "model.joblib"),
        ("../../etc/passwd", "passwd"),
        ("/tmp/model.bin", "model.bin"),
        ("..\\..\\secret.pkl", "secret.pkl"),
        ("", "artifact.bin"),
        ("   ", "artifact.bin"),
        ("/", "artifact.bin"),
    ],
)
def test_safe_input_filename(value: str, expected: str) -> None:
    """Sanitize potentially unsafe upload filenames."""
    assert core.safe_input_filename(value) == expected


def test_run_conversion_dispatches_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch PyTorch requests with all conversion options."""
    seen: dict[str, Path | tuple[int, ...] | int | bool] = {}

    def fake_torch(
        *,
        model_path: Path,
        output_path: Path,
        input_shape: tuple[int, ...],
        opset_version: int,
        allow_unsafe: bool,
    ) -> Path:
        seen["model_path"] = model_path
        seen["output_path"] = output_path
        seen["input_shape"] = input_shape
        seen["opset_version"] = opset_version
        seen["allow_unsafe"] = allow_unsafe
        return output_path

    monkeypatch.setattr(core, "convert_torch_file_to_onnx", fake_torch)
    output = core.run_conversion(
        input_path=Path("m.pt"),
        request=ConversionRequest(
            framework="pytorch",
            filename="m.pt",
            input_shape=(1, 3, 224, 224),
            opset_version=18,
            allow_unsafe=True,
        ),
        output_path=Path("out.onnx"),
    )
    assert output == Path("out.onnx")
    assert seen["input_shape"] == (1, 3, 224, 224)
    assert seen["opset_version"] == 18
    assert seen["allow_unsafe"] is True


def test_run_conversion_dispatches_to_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch TensorFlow requests with opset option."""
    seen: dict[str, Path | int] = {}

    def fake_tf(
        *,
        model_path: Path,
        output_path: Path,
        opset_version: int,
    ) -> Path:
        seen["model_path"] = model_path
        seen["output_path"] = output_path
        seen["opset_version"] = opset_version
        return output_path

    monkeypatch.setattr(core, "convert_tf_path_to_onnx", fake_tf)
    output = core.run_conversion(
        input_path=Path("saved_model"),
        request=ConversionRequest(
            framework="tensorflow",
            filename="saved_model",
            opset_version=17,
        ),
        output_path=Path("out.onnx"),
    )
    assert output == Path("out.onnx")
    assert seen["opset_version"] == 17


def test_run_conversion_dispatches_to_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch scikit-learn requests with safety flag."""
    seen: dict[str, Path | int | bool] = {}

    def fake_sklearn(
        *,
        model_path: Path,
        output_path: Path,
        n_features: int,
        allow_unsafe: bool,
    ) -> Path:
        seen["model_path"] = model_path
        seen["output_path"] = output_path
        seen["n_features"] = n_features
        seen["allow_unsafe"] = allow_unsafe
        return output_path

    monkeypatch.setattr(core, "convert_sklearn_file_to_onnx", fake_sklearn)
    output = core.run_conversion(
        input_path=Path("m.joblib"),
        request=ConversionRequest(
            framework="sklearn",
            filename="m.joblib",
            n_features=5,
            allow_unsafe=True,
        ),
        output_path=Path("out.onnx"),
    )
    assert output == Path("out.onnx")
    assert seen["n_features"] == 5
    assert seen["allow_unsafe"] is True


def test_run_conversion_validates_framework_options() -> None:
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

    with pytest.raises(ValueError, match="unsupported framework"):
        invalid_request = ConversionRequest(framework="tensorflow", filename="m.bin")
        object.__setattr__(invalid_request, "framework", "xgboost")
        core.run_conversion(
            input_path=dummy_path, request=invalid_request, output_path=dummy_output
        )


def test_convert_artifact_bytes_returns_integrity_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify conversion returns stable input/output hashes."""

    def fake_run_conversion(
        input_path: Path,
        request: ConversionRequest,
        output_path: Path,
    ) -> Path:
        assert input_path.name == "model.joblib"
        assert request.framework == "sklearn"
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


def test_convert_artifact_bytes_sanitizes_input_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent path traversal by using a sanitized temp input filename."""

    def fake_run_conversion(
        input_path: Path,
        request: ConversionRequest,
        output_path: Path,
    ) -> Path:
        _ = request
        assert input_path.name == "passwd"
        assert input_path.parent.name.startswith("converter-")
        output_path.write_bytes(b"x")
        return output_path

    monkeypatch.setattr(core, "run_conversion", fake_run_conversion)
    core.convert_artifact_bytes(
        b"payload",
        ConversionRequest(
            framework="tensorflow",
            filename="../../etc/passwd",
        ),
    )


def test_convert_artifact_bytes_rejects_hash_mismatch() -> None:
    """Fail when provided input digest does not match payload."""
    request = ConversionRequest(
        framework="tensorflow",
        filename="saved_model.zip",
        expected_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        core.convert_artifact_bytes(b"actual", request)


def test_convert_artifact_bytes_wraps_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap non-domain exceptions into ConversionError."""

    def fake_run_conversion(
        input_path: Path,
        request: ConversionRequest,
        output_path: Path,
    ) -> Path:
        _ = (input_path, request, output_path)
        raise RuntimeError("boom")

    monkeypatch.setattr(core, "run_conversion", fake_run_conversion)
    with pytest.raises(ConversionError, match="boom"):
        core.convert_artifact_bytes(
            b"payload",
            ConversionRequest(framework="tensorflow", filename="model"),
        )


def test_convert_artifact_bytes_reraises_conversion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate ConversionError without rewriting."""

    def fake_run_conversion(
        input_path: Path,
        request: ConversionRequest,
        output_path: Path,
    ) -> Path:
        _ = (input_path, request, output_path)
        raise ConversionError("bad model")

    monkeypatch.setattr(core, "run_conversion", fake_run_conversion)
    with pytest.raises(ConversionError, match="bad model"):
        core.convert_artifact_bytes(
            b"payload",
            ConversionRequest(framework="tensorflow", filename=""),
        )
