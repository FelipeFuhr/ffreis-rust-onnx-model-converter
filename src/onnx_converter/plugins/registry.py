"""Plugin registry and discovery helpers."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from pydantic import ValidationError

from onnx_converter.errors import PluginError
from onnx_converter.plugins.base import ConverterPlugin, PluginOptions
from onnx_converter.plugins.builtins import SklearnFilePlugin
from onnx_converter.schemas import PluginResolutionConfig


class PluginRegistry:
    """Registry for conversion plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, ConverterPlugin] = {}

    def register(self, plugin: ConverterPlugin) -> None:
        """Register plugin instance by unique name.

        Parameters
        ----------
        plugin : ConverterPlugin
            Plugin instance to register.

        Raises
        ------
        PluginError
            If plugin does not provide a valid name.
        """
        name = getattr(plugin, "name", "").strip()
        if not name:
            raise PluginError("Plugin must define a non-empty 'name'.")
        self._plugins[name] = plugin

    def names(self) -> list[str]:
        """Return registered plugin names.

        Returns
        -------
        list[str]
            Sorted list of plugin names.
        """
        return sorted(self._plugins.keys())

    def get(self, name: str) -> ConverterPlugin:
        """Get plugin by name.

        Parameters
        ----------
        name : str
            Plugin name.

        Returns
        -------
        ConverterPlugin
            Registered plugin instance.

        Raises
        ------
        PluginError
            If plugin name is not registered.
        """
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise PluginError(
                f"Unknown plugin '{name}'. Available plugins: {', '.join(self.names())}"
            ) from exc

    def resolve(
        self,
        model_path: Path,
        model_type: str | None,
        plugin_name: str | None,
        options: PluginOptions,
    ) -> ConverterPlugin:
        """Resolve plugin either explicitly or by ``can_handle`` lookup.

        Parameters
        ----------
        model_path : Path
            Path to source model artifact.
        model_type : str | None
            Optional model family hint.
        plugin_name : str | None
            Explicit plugin name.
        options : Mapping[str, Any]
            Raw plugin options.

        Returns
        -------
        ConverterPlugin
            Resolved plugin.

        Raises
        ------
        PluginError
            If no plugin (or multiple plugins) can handle the request.
        """
        try:
            payload = PluginResolutionConfig(
                model_path=model_path,
                model_type=model_type,
                plugin_name=plugin_name,
                options=dict(options),
            )
        except ValidationError as exc:
            raise PluginError(f"Invalid plugin resolution options: {exc}") from exc

        if payload.plugin_name:
            return self.get(payload.plugin_name)

        matches = [
            plugin
            for plugin in self._plugins.values()
            if plugin.can_handle(
                model_path=payload.model_path,
                model_type=payload.model_type,
                options=payload.options,
            )
        ]
        if not matches:
            raise PluginError(
                "No plugin could handle this model. "
                f"Available plugins: {', '.join(self.names())}"
            )
        if len(matches) > 1:
            names = ", ".join(plugin.name for plugin in matches)
            raise PluginError(
                "Multiple plugins can handle model "
                f"({names}). Pass --plugin-name explicitly."
            )
        return matches[0]

    def load_module(self, module_or_path: str) -> None:
        """Load plugin providers from module name or file path.

        .. warning::
            This method executes code from the specified module. Only load plugins
            from trusted sources.

        Parameters
        ----------
        module_or_path : str
            Python import path or filesystem path to plugin module. Must be from
            a trusted source.
        """
        module = _import_module_or_path(module_or_path)
        _register_from_module(module, self)


def _import_module_or_path(module_or_path: str) -> ModuleType:
    """Import module by import path or filesystem path.

    .. warning::
        This function executes arbitrary Python code from the specified module or file.
        Only load plugins from trusted sources. Malicious plugin code can compromise
        your system security. Plugin loading should only be used with explicit user
        intent (e.g., via ``--plugin-module`` CLI flag) and never with untrusted paths.

    .. note::
        This function intentionally does not restrict file paths to specific directories
        because legitimate plugin use cases require flexibility in plugin locations
        (e.g., project-local plugins, user plugins, system plugins). Security is ensured
        through explicit user action (CLI flags) rather than path restrictions.

    Parameters
    ----------
    module_or_path : str
        Python module path or local file path. Must be from a trusted source.

    Returns
    -------
    ModuleType
        Imported module object.

    Raises
    ------
    PluginError
        If import cannot be completed.
    """
    candidate = Path(module_or_path)
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise PluginError(f"Unable to load plugin module from {candidate}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        return importlib.import_module(module_or_path)
    except Exception as exc:
        raise PluginError(
            f"Unable to import plugin module '{module_or_path}': {exc}"
        ) from exc


def _register_from_module(module: ModuleType, registry: PluginRegistry) -> None:
    """Register plugin definitions found in module.

    Parameters
    ----------
    module : ModuleType
        Imported plugin module.
    registry : PluginRegistry
        Registry that receives plugin objects.
    """
    if hasattr(module, "register_plugins"):
        module.register_plugins(registry)
        return

    plugins_obj = getattr(module, "PLUGINS", None)
    if plugins_obj is not None:
        for plugin in plugins_obj:
            registry.register(plugin)
        return

    plugin_obj = getattr(module, "PLUGIN", None)
    if plugin_obj is not None:
        registry.register(plugin_obj)
        return

    raise PluginError(
        "Plugin module must expose register_plugins(registry), PLUGINS, or PLUGIN."
    )


def create_default_registry(
    extra_modules: Iterable[str] | None = None,
) -> PluginRegistry:
    """Create default plugin registry.

    Parameters
    ----------
    extra_modules : Iterable[str] | None, optional
        Additional plugin modules to load.

    Returns
    -------
    PluginRegistry
        Registry with built-in and external plugins.
    """
    registry = PluginRegistry()
    registry.register(SklearnFilePlugin())
    for module in extra_modules or []:
        registry.load_module(module)
    return registry
