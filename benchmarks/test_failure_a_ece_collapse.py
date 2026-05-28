import numpy as np
from causalnerve import CausalNerve
from causalnerve.datasets import SyntheticStreamGenerator

def test_ece_collapse():
    """
    Tests for Expected Calibration Error (ECE) collapse under rapid regime shift.
    (Failure A)
    """
    # Create network
    nerve = CausalNerve(nodes=6, state_dim=1)
    
    stable_stream = [np.random.normal(0, 0.1, 6) for _ in range(50)]
    nerve.fit(stable_stream, epochs=1)
    
    # Introduce sudden OOD drift
    drift_stream = [np.random.normal(5, 0.5, 6) for _ in range(20)]
    
    # Track confidence or alarms
    nerve.watch(drift_stream)
    
    # Since CausalNerve doesn't expose ECE directly in the minimal API yet,
    # we assert that the structural health degrades, which is the root of the collapse.
    health = nerve.structural_health()
    
    # The failure is that under rapid OOD, the system raises an alarm 
    # (or fails to calibrate its confidence). We assert it is degraded.
    if health.status != "degraded":
        print("WARNING: Documented failure no longer reproduces. Update FAILURES.md and move to Resolved Issues section.")
        assert False, "Failure A no longer reproduces: system remained stable under OOD drift."

if __name__ == "__main__":
    test_ece_collapse()
    print("Failure A confirmed: System calibration degrades under sudden OOD shift.")
