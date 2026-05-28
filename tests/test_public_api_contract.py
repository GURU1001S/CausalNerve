import pytest
import warnings
import numpy as np
import inspect
from causalnerve.api import CausalNerve, CausalNerve
from causalnerve.plugins.interfaces import DomainPlugin, PluginMetadata

class DummyPreset(DomainPlugin):
    @property
    def metadata(self):
        return PluginMetadata(name="dummy", version="1.0")

    def get_nodes(self):
        return {0: {"name": "A"}, 1: {"name": "B"}}
        
    def get_default_edges(self):
        return [(0, 1)]

@pytest.fixture
def nerve():
    return CausalNerve(DummyPreset())

def test_public_api_signatures(nerve):
    """Ensure the 10 core public APIs exist and have proper signatures."""
    core_methods = [
        "fit", "watch", "step", "rollout", "why", "what_if", 
        "do", "predict_next_change", "plot_graph", "export_report"
    ]
    
    for method in core_methods:
        assert hasattr(nerve, method), f"Missing core public API: {method}"
        sig = inspect.signature(getattr(nerve, method))
        assert sig.return_annotation != inspect.Signature.empty, f"{method} is missing return type hint"

def test_api_determinism(nerve):
    """Validate deterministic outputs across core methods."""
    nerve.fit(np.array([[0.5, 0.6], [0.4, 0.5]]))
    
    # 1. Step / Watch
    state = np.array([0.5, 0.5])
    r1 = nerve.step(state)
    r2 = nerve.watch(state)
    assert r1.cycle == 1
    assert r2.cycle == 2
    
    # 2. Rollout
    intervention = {0: 1.0}
    ro1 = nerve.rollout(intervention)
    ro2 = nerve.rollout(intervention)
    assert ro1['peak_divergence'] == ro2['peak_divergence']
    
    # 3. Do
    do_res = nerve.do(0, 1.0)
    assert do_res['new_value'] == 1.0
    
    # 4. What If
    wi = nerve.what_if(0, 1.0)
    assert isinstance(wi['confidence'], float)

def test_backward_compatibility_deprecation(nerve):
    """Ensure run_counterfactual() throws a DeprecationWarning but still works."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = nerve.run_counterfactual({0: 1.0})
        
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "run_counterfactual" in str(w[-1].message)
        assert "rollout()" in str(w[-1].message)
        
        # Must return same format as rollout
        assert "peak_divergence" in res
