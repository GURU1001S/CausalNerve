import importlib
import inspect
import pkgutil
from typing import Dict, Type, Optional, Any
from causalnerve.plugins.interfaces import (
    BasePlugin, DomainPlugin, ConstraintPlugin, 
    VisualizerPlugin, ReasonerPlugin
)

class PluginRegistry:
    """Central registry for discovering and loading CausalNerve plugins."""
    _domains: Dict[str, DomainPlugin] = {}
    _constraints: Dict[str, ConstraintPlugin] = {}
    _visualizers: Dict[str, VisualizerPlugin] = {}
    _reasoners: Dict[str, ReasonerPlugin] = {}

    @classmethod
    def register(cls, plugin: BasePlugin):
        """Register an instantiated plugin based on its interfaces."""
        name = plugin.metadata.name
        
        if isinstance(plugin, DomainPlugin):
            if name in cls._domains:
                raise ValueError(f"Domain plugin {name} already registered.")
            cls._domains[name] = plugin
            
        if isinstance(plugin, ConstraintPlugin):
            cls._constraints[name] = plugin
            
        if isinstance(plugin, VisualizerPlugin):
            cls._visualizers[name] = plugin
            
        if isinstance(plugin, ReasonerPlugin):
            cls._reasoners[name] = plugin

    @classmethod
    def get_domain(cls, name: str) -> Optional[DomainPlugin]:
        return cls._domains.get(name)

    @classmethod
    def get_constraint_engine(cls, name: str) -> Optional[ConstraintPlugin]:
        return cls._constraints.get(name)

    @classmethod
    def auto_discover(cls):
        """Discovers standard plugins in the causalnerve.domains namespace."""
        try:
            import causalnerve.plugins
            for _, module_name, _ in pkgutil.iter_modules(causalnerve.plugins.__path__):
                mod = importlib.import_module(f"causalnerve.plugins.{module_name}.plugin")
                for item_name in dir(mod):
                    item = getattr(mod, item_name)
                    if inspect.isclass(item) and issubclass(item, BasePlugin) and item != BasePlugin:
                        # Instantiate and register
                        try:
                            cls.register(item())
                        except (TypeError, NotImplementedError):
                            pass # Abstract classes
        except ImportError:
            pass

    @classmethod
    def clear(cls):
        """Clear all registered plugins."""
        cls._domains.clear()
        cls._constraints.clear()
        cls._visualizers.clear()
        cls._reasoners.clear()
