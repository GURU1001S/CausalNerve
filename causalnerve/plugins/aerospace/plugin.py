from typing import Dict, Any, List
from causalnerve.plugins.interfaces import DomainPlugin, PluginMetadata

class AerospaceDomain(DomainPlugin):
    """NASA C-MAPSS FD004 Turbofan Engine Domain Knowledge."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="aerospace",
            version="1.0.0",
            dependencies=[],
            capabilities=["node_resolution", "thermodynamic_grounding"],
            citation="NASA C-MAPSS Trajectory Data"
        )
        
    def get_nodes(self) -> Dict[int, Dict[str, Any]]:
        return {
            0: {"name": "Fan", "short": "FAN", "subsystem": "inlet"},
            1: {"name": "LPC", "short": "LPC", "subsystem": "compression"},
            2: {"name": "HPC", "short": "HPC", "subsystem": "compression"},
            3: {"name": "Combustor", "short": "CMB", "subsystem": "combustion"},
            4: {"name": "HPT", "short": "HPT", "subsystem": "turbine"},
            5: {"name": "LPT", "short": "LPT", "subsystem": "turbine"},
            6: {"name": "H.Spool", "short": "HPS", "subsystem": "mechanical"},
            7: {"name": "L.Spool", "short": "LPS", "subsystem": "mechanical"},
            8: {"name": "P.Bank", "short": "PBK", "subsystem": "pressure"},
            9: {"name": "Cooling", "short": "CLG", "subsystem": "thermal"},
            10: {"name": "Bypass", "short": "BYP", "subsystem": "flow"},
            11: {"name": "Fuel", "short": "FUEL", "subsystem": "fuel"},
            12: {"name": "Snsr.A", "short": "S_A", "subsystem": "sensor"},
            13: {"name": "Snsr.B", "short": "S_B", "subsystem": "sensor"},
        }

    def get_default_edges(self) -> List[tuple]:
        return [
            (11, 3), (3, 4), (4, 2), (4, 6), (6, 2), (2, 1), 
            (5, 7), (7, 0), (9, 4), (10, 1), (4, 12), (3, 12)
        ]
