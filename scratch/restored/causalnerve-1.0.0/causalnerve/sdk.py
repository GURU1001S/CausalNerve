from typing import Optional
from causalnerve.plugins.registry import PluginRegistry
from causalnerve.plugins.interfaces import DomainPlugin

class CausalNerve:
    """Main SDK entry point for initializing CausalNerve domains."""
    
    @classmethod
    def from_preset(cls, domain_name: str) -> 'CausalNerveInstance':
        """Initialize the SDK with a specific domain preset."""
        PluginRegistry.auto_discover()
        domain = PluginRegistry.get_domain(domain_name)
        
        if not domain:
            raise ValueError(f"Domain '{domain_name}' not found. Available domains: {list(PluginRegistry._domains.keys())}")
            
        return CausalNerveInstance(domain)

class CausalNerveInstance:
    """An initialized instance of the SDK tailored to a domain."""
    def __init__(self, domain: DomainPlugin):
        self.domain = domain
        self.nodes = domain.get_nodes()
        self.edges = domain.get_default_edges()
