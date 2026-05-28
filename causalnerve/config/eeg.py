"""
causalnerve.config.eeg
=======================
EEG brain connectivity preset.
"""

from .base import CausalPreset

class EEGPreset(CausalPreset):
    """
    EEG brain connectivity preset.
    Models directed functional connectivity between 
    EEG electrode regions.
    """
    
    name = "eeg"
    n_nodes = 19
    default_persistence = 0.70    # faster dynamics than turbofan
    alarm_threshold = 0.08
    
    # 19 standard 10-20 electrode positions
    node_labels = {
        0: "Fp1", 1: "Fp2", 2: "F7", 3: "F3",
        4: "Fz", 5: "F4", 6: "F8", 7: "T7",
        8: "C3", 9: "Cz", 10: "C4", 11: "T8",
        12: "P7", 13: "P3", 14: "Pz", 15: "P4",
        16: "P8", 17: "O1", 18: "O2"
    }
    
    def plausibility_fn(self, src, dst, state):
        # Ipsilateral connections more plausible than contralateral
        # (simplified heuristic — users can override)
        return True  # conservative default: all allowed
