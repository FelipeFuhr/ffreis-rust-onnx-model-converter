"""Shared conversion-daemon core utilities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from onnx_converter.api import (
    convert_sklearn_file_to_onnx,
    convert_tf_path_to_onnx,
    convert_torch_file_to_onnx,
)
from onnx_converter.errors import ConversionError
from onnx_converter.types import FrameworkKind

Framework = FrameworkKind


@dataclass(frozen=True)
class ConversionRequest:
    """Normalized conversion request.

    Parameters
    ----------
    framework : {"pytorch", "tensorflow", "sklearn"}
        Source artifact framework.
    filename : str
        Original artifact filename.
    expected_sha256 : str | None, default=None
        Optional expected SHA-256 digest of input artifact.
    input_shape : tuple[int, ...] | None, default=None
        Required for PyTorch conversion.
    n_features : int | None, default=None
        Required for scikit-learn conversion.
    opset_version : int, default=14
        ONNX opset version.
    allow_unsafe : bool, default=False
        Allow unsafe pickle loading for frameworks that support it.
    """

    framework: Framework
    filename: str
    expected_sha256: str | None = None
    input_shape: tuple[int, ...] | None = None
    n_features: int | None = None
    opset_version: int = 14
    allow_unsafe: bool = False


@dataclass(frozen=True)
class ConversionOutcome:
    """Conversion output metadata."""

    output_bytes: bytes
    output_filename: str
    output_sha256: str
    output_size_bytes: int


def digest_bytes(data: bytes) -> str:
    """Compute SHA-256 digest for byte payload."""
    return sha256(data).hexdigest()


def digest_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 digest for file content."""
    hasher = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_sha256(value: str | None) -> str | None:
    """Normalize and validate SHA-256 string if provided."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError("expected_sha256 must be a 64-character hex digest")
    return normalized


def write_bytes_to_file(path: Path, data: bytes) -> str:
    """Write bytes to file and return SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest_bytes(data)


def safe_input_filename(filename: str) -> str:
    """Return a filesystem-safe artifact filename for temp-dir writes."""
    raw = filename.strip()
    if not raw:
        return "artifact.bin"
    # Normalize Windows-style separators before basename extraction.
    normalized = raw.replace("\\", "/")
    candidate = Path(normalized).name
    if candidate in {"", ".", ".."}:
        return "artifact.bin"
    return candidate


def run_conversion(
    input_path: Path,
    request: ConversionRequest,
    output_path: Path,
) -> Path:
    """Run framework-specific conversion against a local artifact path."""
    if request.framework == "pytorch":
        if not request.input_shape:
            raise ValueError("input_shape is required for pytorch conversion")
        return convert_torch_file_to_onnx(
            model_path=input_path,
            output_path=output_path,
            input_shape=request.input_shape,
            opset_version=request.opset_version,
            allow_unsafe=request.allow_unsafe,
        )
    if request.framework == "tensorflow":
        return convert_tf_path_to_onnx(
            model_path=input_path,
            output_path=output_path,
            opset_version=request.opset_version,
        )
    if request.framework == "sklearn":
        if request.n_features is None or request.n_features <= 0:
            raise ValueError("n_features must be > 0 for sklearn conversion")
        return convert_sklearn_file_to_onnx(
            model_path=input_path,
            output_path=output_path,
            n_features=request.n_features,
            allow_unsafe=request.allow_unsafe,
        )
    raise ValueError(f"unsupported framework: {request.framework}")


def convert_artifact_bytes(
    data: bytes,
    request: ConversionRequest,
) -> tuple[str, ConversionOutcome]:
    """Convert artifact bytes and return input/output integrity metadata.

    Raises
    ------
    ValueError
        If request validation fails.
    ConversionError
        If conversion fails.
    """
    expected_sha = normalize_sha256(request.expected_sha256)
    with TemporaryDirectory(prefix="converter-") as tmp:
        tmp_dir = Path(tmp)
        input_name = safe_input_filename(request.filename)
        input_path = tmp_dir / input_name
        input_sha = write_bytes_to_file(input_path, data)
        if expected_sha is not None and input_sha != expected_sha:
            raise ValueError("input SHA-256 mismatch")

        output_path = tmp_dir / f"{input_path.stem}.onnx"
        try:
            actual_output = run_conversion(
                input_path=input_path,
                request=request,
                output_path=output_path,
            )
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(str(exc)) from exc

        output_bytes = actual_output.read_bytes()
        output_sha = digest_bytes(output_bytes)
        outcome = ConversionOutcome(
            output_bytes=output_bytes,
            output_filename=actual_output.name,
            output_sha256=output_sha,
            output_size_bytes=len(output_bytes),
        )
        return input_sha, outcome
