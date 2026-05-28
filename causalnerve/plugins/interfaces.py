from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class PluginMetadata:
    name: str
    version: str
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    citation: str = ""

class BasePlugin(ABC):
    @property
    def metadata(self) -> PluginMetadata:
        raise NotImplementedError

class DomainPlugin(BasePlugin):
    """Provides semantic knowledge about a specific domain (nodes, rules)."""
    def get_nodes(self) -> Dict[int, Dict[str, Any]]:
        raise NotImplementedError
    
    def get_default_edges(self) -> List[tuple]:
        return []

class ConstraintPlugin(BasePlugin):
    """Evaluates the thermodynamic/physical plausibility of edges."""
    def evaluate_edge(self, src: int, dst: int, conf: float) -> float:
        raise NotImplementedError

class VisualizerPlugin(BasePlugin):
    """Provides custom dashboard layouts or widgets."""
    def get_ui_schema(self) -> Dict[str, Any]:
        return {}

class ReasonerPlugin(BasePlugin):
    """Translates graphs to semantic macro-states."""
    def process_graph(self, edge_matrix: List[List[float]]) -> Dict[str, Any]:
        return {}
