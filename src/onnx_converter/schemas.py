"""Pydantic schemas for runtime validation of conversion inputs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onnx_converter.types import OptionValue


class TorchFileConversionConfig(BaseModel):
    """Validated input for file-based PyTorch conversion."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    output_path: Path
    input_shape: tuple[int, ...]
    opset_version: int = Field(default=14, ge=1)
    allow_unsafe: bool = False

    @field_validator("input_shape")
    @classmethod
    def _validate_input_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("input_shape must contain at least one dimension.")
        if any(dim <= 0 for dim in value):
            raise ValueError("input_shape dimensions must be positive integers.")
        return value


class TensorflowFileConversionConfig(BaseModel):
    """Validated input for file-based TensorFlow conversion."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    output_path: Path
    opset_version: int = Field(default=14, ge=1)


class SklearnFileConversionConfig(BaseModel):
    """Validated input for file-based scikit-learn conversion."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    output_path: Path
    n_features: int = Field(gt=0)
    allow_unsafe: bool = False


class PytorchConversionConfig(BaseModel):
    """Validated input for in-memory PyTorch ONNX export."""

    model_config = ConfigDict(extra="forbid")

    output_path: Path
    input_shape: tuple[int, ...]
    input_names: list[str] = Field(default_factory=lambda: ["input"])
    output_names: list[str] = Field(default_factory=lambda: ["output"])
    dynamic_axes: dict[str, dict[int, str]] | None = None
    opset_version: int = Field(default=14, ge=1)

    @field_validator("input_shape")
    @classmethod
    def _validate_export_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("input_shape must contain at least one dimension.")
        if any(dim <= 0 for dim in value):
            raise ValueError("input_shape dimensions must be positive integers.")
        return value

    @field_validator("input_names", "output_names")
    @classmethod
    def _validate_names(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("input/output names cannot be empty.")
        if any(not item.strip() for item in value):
            raise ValueError("input/output names cannot contain empty entries.")
        return value


class SklearnConversionConfig(BaseModel):
    """Validated input for in-memory sklearn ONNX export."""

    model_config = ConfigDict(extra="forbid")

    output_path: Path
    target_opset: int | None = Field(default=None, ge=1)
    initial_types: list[tuple[str, object]] | None = None


class SklearnPluginOptions(BaseModel):
    """Validated options for the built-in sklearn plugin."""

    model_config = ConfigDict(extra="ignore", strict=True)

    n_features: int = Field(gt=0)
    allow_unsafe: bool = False
    optimize: bool = False
    quantize_dynamic: bool = False
    opset_version: int | None = Field(default=None, ge=1)
    metadata: dict[str, str] | None = None
    parity_input_path: Path | None = None
    parity_atol: float = Field(default=1e-5, gt=0.0)
    parity_rtol: float = Field(default=1e-4, gt=0.0)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: OptionValue) -> dict[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("metadata must be a mapping.")
        return {str(key): str(item) for key, item in value.items()}


class PluginResolutionConfig(BaseModel):
    """Validated input for plugin registry resolution."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    model_type: str | None = None
    plugin_name: str | None = None
    options: dict[str, object] = Field(default_factory=dict)
