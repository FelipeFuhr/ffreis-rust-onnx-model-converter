"""Application-layer result objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConversionResult:
    """Structured conversion outcome."""

    output_path: Path
    framework: str
    source_path: Path
    metadata: Mapping[str, str] | None = None
