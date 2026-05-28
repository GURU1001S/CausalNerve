import numpy as np
from causalnerve import CausalNerve
import copy

def test_dropout_breakdown():
    """
    Tests breakdown under severe sensor dropout (>30%).
    (Failure D)
    """
    nerve = CausalNerve(nodes=5)
    
    stream = [np.random.normal(0, 0.1, 5) for _ in range(50)]
    nerve.fit(stream, epochs=5)
    
    # Introduce >35% dropout (NaNs or zeros)
    drift = []
    for _ in range(50):
        x = np.random.normal(0, 0.1, 5)
        # Randomly drop 2 out of 5 sensors (~40%)
        drop_idx = np.random.choice(5, size=2, replace=False)
        x[drop_idx] = 0.0 # Or np.nan if supported
        drift.append(x)
        
    nerve.watch(drift)
    
    health = nerve.structural_health()
    # At 40% dropout, the system should fail to maintain stable predictions
    # and trigger degraded status.
    if health.status != "degraded":
        print("WARNING: Documented failure no longer reproduces. Update FAILURES.md and move to Resolved Issues section.")
        assert False, "Failure D no longer reproduces: system handled 40% dropout perfectly."

if __name__ == "__main__":
    test_dropout_breakdown()
    print("Failure D confirmed: System degrades under severe sensor dropout.")
