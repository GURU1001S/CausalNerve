import inspect
from causalnerve import CausalNerve

def test_rollout_signature():
    sig = inspect.signature(CausalNerve.rollout)
    
    # Assert parameters exist
    assert "intervention" in sig.parameters
    assert "horizon" in sig.parameters
    assert "steps" in sig.parameters
    
    # Assert backwards compatibility: steps=None, horizon=50
    assert sig.parameters["horizon"].default == 50
    assert sig.parameters["steps"].default is None
    
    print("Rollout signature contract passed.")
