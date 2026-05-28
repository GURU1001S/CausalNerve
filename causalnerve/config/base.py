"""
causalnerve.config.base
========================
Base class for all domain presets.
A preset is a bundle of domain knowledge that configures
a CausalNerve instance for a specific domain.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class CausalPreset:
    """
    Base class for all domain presets.
    Users can create their own presets by subclassing this.
    """
    
    # Required attributes
    name: str = "base"
    n_nodes: int = 0
    node_labels: Dict[int, str] = {}
    default_edges: List[Tuple[int, int]] = []
    default_persistence: float = 0.90
    state_variables: List[str] = []
    
    # Optional domain knowledge
    plausibility_rules: Dict = {}
    thermal_regimes: Optional[Dict] = None
    alarm_threshold: float = 0.05
    
    def plausibility_fn(self, src: int, dst: int, state: Any) -> bool:
        """
        Returns True if edge (src,dst) is physically plausible
        given current state.
        """
        return True
    
    def configure(self, nerve: Any) -> Any:
        """
        Apply this preset to a CausalNerve instance.
        Sets nodes, edges, labels, persistence, plausibility.
        Called by CausalNerve.from_preset("domain").
        """
        nerve.n_nodes = self.n_nodes
        nerve.node_labels = self.node_labels
        nerve.known_edges = self.default_edges
        nerve.persistence = self.default_persistence
        nerve.alarm_threshold = self.alarm_threshold
        # The plausibility_fn can be passed to the OCGR graph surgery engine
        nerve.plausibility_fn = self.plausibility_fn
        return nerve
