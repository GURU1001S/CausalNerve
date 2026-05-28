"""
causalnerve.config.turbofan
============================
Turbofan engine digital twin preset.
Based on NASA C-MAPSS sensor suite.
"""

from .base import CausalPreset

class TurbofanPreset(CausalPreset):
    """
    Turbofan engine digital twin preset.
    Derived from validated SE-OCGR experimental results.
    """
    
    name = "turbofan"
    n_nodes = 14
    default_persistence = 0.90
    alarm_threshold = 0.05
    
    node_labels = {
        0: "Fan",
        1: "LPC",
        2: "HPC",
        3: "Combustor",
        4: "HPT",
        5: "LPT",
        6: "HP_Spool",
        7: "LP_Spool",
        8: "Pressure_Bank",
        9: "Cooling",
        10: "Bypass",
        11: "Fuel",
        12: "Sensor_A",
        13: "Sensor_B"
    }
    
    default_edges = [
        (0,1), (1,2), (2,3), (3,4),
        (4,5), (6,2), (7,0), (11,3)
    ]
    
    # Thermodynamic plausibility rules
    ALWAYS_IMPLAUSIBLE = [
        (13, 0),   # Demand signal → Fan physics: impossible
        (10, 3),   # Bypass → Combustor: bypass never enters combustor
        (9, 0),    # Cooling → Fan: coolant isolated from fan
    ]
    
    HIGH_THERMAL_PLAUSIBLE = [
        (4, 2),    # HPT → HPC: thermal feedback at high T30
        (3, 4),    # Combustor → HPT: hot gas coupling
        (2, 9),    # HPC → Cooling: bleed air interaction
    ]
    
    def plausibility_fn(self, src, dst, state):
        if (src, dst) in self.ALWAYS_IMPLAUSIBLE:
            return False
        # State-conditioned plausibility
        # T30 is typically index 2 (HPC)
        try:
            T30_norm = state[2] if len(state) > 2 else 0.5
        except (TypeError, IndexError):
            T30_norm = 0.5
            
        if T30_norm > 0.80:
            return (src, dst) in self.HIGH_THERMAL_PLAUSIBLE
        return True
