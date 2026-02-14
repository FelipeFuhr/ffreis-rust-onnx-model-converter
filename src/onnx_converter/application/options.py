"""Typed option objects shared across conversion use-cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParityOptions:
    """Parity check configuration."""

    input_path: Path | None = None
    atol: float = 1e-5
    rtol: float = 1e-4


@dataclass(frozen=True)
class PostprocessOptions:
    """ONNX post-processing configuration."""

    optimize: bool = False
    quantize_dynamic: bool = False
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ConversionOptions:
    """Shared conversion options passed through use-cases."""

    allow_unsafe: bool = False
    opset_version: int = 14
    parity: ParityOptions = ParityOptions()
    postprocess: PostprocessOptions = PostprocessOptions()
