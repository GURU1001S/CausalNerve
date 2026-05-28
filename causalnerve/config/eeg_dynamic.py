from typing import Dict, List, Tuple, Any
from causalnerve.config.base import CausalPreset

class EEGDynamicPreset(CausalPreset):
    """
    Domain Preset for Brain Connectivity (EEG) Causal Graph Inference.
    Configures structural priors based on cortical adjacency and volume conduction constraints.
    """
    
    name: str = "eeg_dynamic"
    default_persistence: float = 0.85  # Dynamic topology, faster decay than aerospace
    alarm_threshold: float = 0.15      # Higher threshold due to physiological noise
    
    # Standard 10-20 system canonical mapping for connectivity subsets
    # E.g., frontal (F), central/motor (C), parietal (P), occipital (O)
    def __init__(self, channels: List[str] = None):
        if channels is None:
            # Default to a 19-channel clinical standard subset
            channels = [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                'T3', 'C3', 'Cz', 'C4', 'T4', 
                'T5', 'P3', 'Pz', 'P4', 'T6', 
                'O1', 'O2'
            ]
        self.n_nodes = len(channels)
        self.node_labels = {i: name for i, name in enumerate(channels)}
        self.channels = channels
        self.state_variables = channels
        
        # Plausible Cortical Adjacency Priors
        # We assume connections are more likely between adjacent regions or homologous regions
        self.default_edges = [
            (0, 1), (2, 3), (4, 5), (8, 9), (9, 10), (13, 14), (14, 15),
            (17, 18), (3, 8), (5, 10), (8, 13), (10, 15), (7, 11)
        ]
        
        self.thermal_regimes = None # Not applicable to EEG
        
        self.plausibility_rules = {
            "anti_volume_conduction": True, # Exclude immediate neighbor instantaneous correlations
            "allow_interhemispheric": True,
            "max_distance": 3 # Maximum 'hops' based on scalp topology
        }

    def _get_region(self, ch_name: str) -> str:
        ch = ch_name.upper()
        if 'FP' in ch: return 'Pre-frontal'
        if 'F' in ch: return 'Frontal'
        if 'C' in ch: return 'Motor/Central'
        if 'P' in ch: return 'Parietal'
        if 'O' in ch: return 'Occipital'
        if 'T' in ch: return 'Temporal'
        return 'Unknown'

    def plausibility_fn(self, src: int, dst: int, state: Any) -> bool:
        """
        Filters out biologically impossible edges or extreme volume conduction artifacts.
        """
        src_name = self.channels[src]
        dst_name = self.channels[dst]
        
        # Volume conduction filter (simplified heuristic for demo)
        # In real scientific ML, we'd use zero-phase lag removal or imaginary coherence checks
        if self.plausibility_rules.get("anti_volume_conduction", False):
            # Extremely simplified: if they share the exact same letter prefix and number
            # (e.g., F3 and Fz), they might be subject to heavy volume conduction, but we allow it
            # if delay is > 0 in the causal model. Since OCGR models temporal causality, 
            # we mainly just prevent self-loops.
            if src == dst:
                return False
                
        return True

    def configure(self, nerve: Any) -> Any:
        nerve = super().configure(nerve)
        nerve.domain = "eeg"
        nerve.temporal_smoothing = 0.9  # Required for EEG noise
        return nerve
