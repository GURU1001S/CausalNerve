import os
from typing import Dict, Any, List
from causalnerve.plugins.interfaces import DomainPlugin, PluginMetadata
from causalnerve.config.eeg_dynamic import EEGDynamicPreset

class EEGDomainPlugin(DomainPlugin):
    """
    CausalNerve EEG Domain Plugin.
    Provides canonical 10-20 system nodes and biological priors.
    """
    
    def __init__(self):
        self.preset = EEGDynamicPreset()
        
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="eeg",
            version="1.0.0",
            capabilities=["dynamic_topology", "biology_priors"]
        )

    def get_nodes(self) -> Dict[int, Dict[str, Any]]:
        nodes = {}
        for idx, name in self.preset.node_labels.items():
            nodes[idx] = {
                "name": name,
                "short": name,
                "region": self.preset._get_region(name)
            }
        return nodes
        
    def get_default_edges(self) -> List[tuple]:
        return self.preset.default_edges
