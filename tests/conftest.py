"""Shared pytest configuration and marker assignment."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Attach suite markers based on test file path."""
    del config
    for item in items:
        path = Path(str(item.fspath))
        parts = set(path.parts)
        if "e2e_tests" in parts:
            item.add_marker(pytest.mark.e2e)
        elif "integration_tests" in parts:
            item.add_marker(pytest.mark.integration)
        elif "unit_tests" in parts:
            item.add_marker(pytest.mark.unit)
