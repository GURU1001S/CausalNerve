"""
causalnerve.config.climate
===========================
Climate subsystem causal modeling preset.
"""

from .base import CausalPreset

class ClimatePreset(CausalPreset):
    """
    Climate subsystem causal modeling preset.
    Use case: model causal couplings between climate subsystems,
    simulate intervention effects, detect regime shifts.
    """
    
    name = "climate"
    n_nodes = 8
    default_persistence = 0.98    # very slow dynamics
    alarm_threshold = 0.02        # tight — climate shifts are slow
    
    node_labels = {
        0: "AtmosphericCO2",
        1: "OceanHeatContent",
        2: "ArcticSeaIce",
        3: "GlobalSurfaceTemp",
        4: "PrecipitationPattern",
        5: "VegetationCover",
        6: "MethaneConcentration",
        7: "OceanCirculation"
    }
    
    default_edges = [
        (0,3), (3,2), (2,1), (1,3),
        (3,4), (4,5), (6,3), (1,7)
    ]
