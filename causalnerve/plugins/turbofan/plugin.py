from typing import Dict, Any, List
from causalnerve.plugins.interfaces import DomainPlugin, PluginMetadata

class TurbofanDomain(DomainPlugin):
    """NASA C-MAPSS FD001 Turbofan Engine Domain Knowledge."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="turbofan",
            version="1.0.0",
            dependencies=[],
            capabilities=["node_resolution", "thermodynamic_grounding"],
            citation="NASA C-MAPSS Trajectory Data"
        )
        
    def get_nodes(self) -> Dict[int, Dict[str, Any]]:
        return {i: {"name": f"S{i}", "short": f"S{i}", "subsystem": "sensor"} for i in range(21)}

    def get_default_edges(self) -> List[tuple]:
        # A plausible ground truth graph for the 21 sensors
        return [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)]
