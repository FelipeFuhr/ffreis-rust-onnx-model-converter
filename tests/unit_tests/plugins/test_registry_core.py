"""Unit tests for plugin registry resolution and module loading helpers."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from onnx_converter.errors import PluginError
from onnx_converter.plugins.registry import (
    PluginRegistry,
    _import_module_or_path,
    _register_from_module,
    create_default_registry,
)


class _Plugin:
    """Simple plugin test double."""

    def __init__(self, name: str, handles: bool = True) -> None:
        self.name = name
        self._handles = handles

    def can_handle(self, model_path: Path, model_type: str | None) -> bool:
        del model_path, model_type
        return self._handles

    def convert(self, model_path: Path, output_path: Path, options: object) -> Path:
        del model_path, options
        return output_path


def test_register_requires_non_empty_name() -> None:
    """Reject plugins without a non-empty name."""
    registry = PluginRegistry()
    with pytest.raises(PluginError, match="non-empty 'name'"):
        registry.register(_Plugin(name="  "))


def test_get_unknown_plugin_raises() -> None:
    """Raise clear error for unknown plugin lookup."""
    registry = PluginRegistry()
    with pytest.raises(PluginError, match="Unknown plugin"):
        registry.get("missing")


def test_resolve_by_explicit_name() -> None:
    """Resolve explicit plugin_name without can_handle scan."""
    registry = PluginRegistry()
    plugin = _Plugin("x", handles=False)
    registry.register(plugin)
    out = registry.resolve(Path("m.bin"), None, "x", {})
    assert out is plugin


def test_resolve_no_matches_raises() -> None:
    """Raise when no registered plugin can handle the model."""
    registry = PluginRegistry()
    registry.register(_Plugin("x", handles=False))
    with pytest.raises(PluginError, match="No plugin could handle"):
        registry.resolve(Path("m.bin"), None, None, {})


def test_resolve_multiple_matches_raises() -> None:
    """Raise when multiple plugins match and no explicit name is provided."""
    registry = PluginRegistry()
    registry.register(_Plugin("a", handles=True))
    registry.register(_Plugin("b", handles=True))
    with pytest.raises(PluginError, match="Multiple plugins can handle"):
        registry.resolve(Path("m.bin"), None, None, {})


def test_resolve_returns_single_match() -> None:
    """Return the sole matching plugin when exactly one plugin can handle."""
    registry = PluginRegistry()
    plugin = _Plugin("single", handles=True)
    registry.register(plugin)
    assert registry.resolve(Path("m.bin"), None, None, {}) is plugin


def test_resolve_wraps_validation_errors() -> None:
    """Wrap pydantic validation errors as PluginError."""
    registry = PluginRegistry()
    with pytest.raises(PluginError, match="Invalid plugin resolution options"):
        registry.resolve(None, None, None, {})  # type: ignore[arg-type]


def test_import_module_by_path_and_register_variants(tmp_path: Path) -> None:
    """Load plugin module from file path and register via supported contracts."""
    plugin_file = tmp_path / "plugin_mod.py"
    plugin_file.write_text(
        "class P:\n"
        "    name='p'\n"
        "    def can_handle(self, model_path, model_type):\n"
        "        return True\n"
        "    def convert(self, model_path, output_path, options):\n"
        "        return output_path\n"
        "PLUGIN = P()\n",
        encoding="utf-8",
    )
    module = _import_module_or_path(str(plugin_file))
    registry = PluginRegistry()
    _register_from_module(module, registry)
    assert registry.get("p").name == "p"


def test_import_module_invalid_path_spec_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise PluginError when file path exists but import spec is invalid."""
    plugin_file = tmp_path / "plugin_mod.py"
    plugin_file.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "onnx_converter.plugins.registry.importlib.util.spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(PluginError, match="Unable to load plugin module"):
        _import_module_or_path(str(plugin_file))


def test_import_module_by_name_failure_raises() -> None:
    """Raise PluginError when import path cannot be imported."""
    with pytest.raises(PluginError, match="Unable to import plugin module"):
        _import_module_or_path("module.that.does.not.exist")


def test_register_from_module_uses_register_plugins() -> None:
    """Prefer register_plugins(registry) hook when available."""
    registry = PluginRegistry()
    module = types.SimpleNamespace(
        register_plugins=lambda r: r.register(_Plugin("hook"))
    )
    _register_from_module(module, registry)
    assert registry.get("hook").name == "hook"


def test_register_from_module_with_plugins_list() -> None:
    """Register all plugins from PLUGINS iterable contract."""
    registry = PluginRegistry()
    module = types.SimpleNamespace(PLUGINS=[_Plugin("a"), _Plugin("b")])
    _register_from_module(module, registry)
    assert registry.names() == ["a", "b"]


def test_register_from_module_requires_contract() -> None:
    """Raise when plugin module exposes no supported registration contract."""
    with pytest.raises(PluginError, match="must expose"):
        _register_from_module(types.SimpleNamespace(), PluginRegistry())


def test_registry_load_module_calls_import_and_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute load_module wrapper path through helper functions."""
    registry = PluginRegistry()
    module = types.SimpleNamespace(PLUGIN=_Plugin("x"))
    monkeypatch.setattr(
        "onnx_converter.plugins.registry._import_module_or_path", lambda _path: module
    )
    registry.load_module("pkg.mod")
    assert "x" in registry.names()


def test_create_default_registry_loads_extra_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load extra plugin modules passed into create_default_registry."""
    loaded: list[str] = []

    def fake_load_module(self: PluginRegistry, module: str) -> None:
        loaded.append(module)

    monkeypatch.setattr(PluginRegistry, "load_module", fake_load_module)
    registry = create_default_registry(extra_modules=["a.b", "c.d"])
    assert "sklearn_file" in registry.names()
    assert loaded == ["a.b", "c.d"]
