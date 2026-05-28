import pytest
from causalnerve.plugins.registry import PluginRegistry
from causalnerve.plugins.interfaces import DomainPlugin
from causalnerve.api import CausalNerve

def test_plugin_auto_discovery():
    PluginRegistry.clear()
    PluginRegistry.auto_discover()
    
    # Check if aerospace domain is loaded
    domain = PluginRegistry.get_domain("aerospace")
    assert domain is not None
    assert isinstance(domain, DomainPlugin)
    assert domain.metadata.name == "aerospace"

def test_domain_nodes():
    PluginRegistry.clear()
    PluginRegistry.auto_discover()
    domain = PluginRegistry.get_domain("aerospace")
    nodes = domain.get_nodes()
    
    assert 0 in nodes
    assert nodes[0]["name"] == "Fan"
    assert len(nodes) == 14

def test_sdk_from_preset():
    PluginRegistry.clear()
    instance = CausalNerve.from_preset("aerospace")
    assert instance.domain.metadata.name == "aerospace"
    assert len(instance.nodes) == 14

def test_invalid_preset():
    PluginRegistry.clear()
    with pytest.raises(ValueError):
        CausalNerve.from_preset("unknown_domain")
