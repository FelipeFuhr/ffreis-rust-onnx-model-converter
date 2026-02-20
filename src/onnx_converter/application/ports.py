"""Application ports for clean architecture boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from onnx_converter.application.options import ParityOptions, PostprocessOptions
from onnx_converter.types import ModelArtifact, OptionMap


class ModelLoader(Protocol):
    """Load model artifact into an in-memory representation."""

    def load(self, model_path: Path, allow_unsafe: bool = False) -> ModelArtifact:
        """Load model from path."""


class ModelConverter(Protocol):
    """Convert in-memory model into ONNX."""

    def convert(
        self,
        model: ModelArtifact,
        output_path: Path,
        options: OptionMap,
    ) -> Path:
        """Convert model and return ONNX path."""


class ParityChecker(Protocol):
    """Check numerical parity between source model and ONNX."""

    def check(
        self,
        model: ModelArtifact,
        onnx_path: Path,
        parity: ParityOptions,
        context: OptionMap | None = None,
    ) -> None:
        """Raise on mismatch."""


class OnnxPostProcessor(Protocol):
    """Apply metadata/optimization/quantization to ONNX artifact."""

    def run(
        self,
        output_path: Path,
        source_path: Path,
        framework: str,
        config_metadata: Mapping[str, str],
        options: PostprocessOptions,
    ) -> None:
        """Apply configured post-processing in place."""
