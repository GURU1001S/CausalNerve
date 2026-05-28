import numpy as np
from causalnerve import CausalNerve

def test_lyapunov_local_minima():
    """
    Tests Lyapunov Convergence Traps.
    (Failure C)
    """
    nerve = CausalNerve(nodes=3)
    
    # Create a cyclic dependency which forces a local minimum
    stream = []
    for _ in range(100):
        x = np.random.normal(0, 0.1, 3)
        # 0 -> 1 -> 2 -> 0 loop
        x[1] += 0.8 * x[0]
        x[2] += 0.8 * x[1]
        x[0] += 0.8 * x[2]
        stream.append(x)
        
    nerve.fit(stream[:50], epochs=5)
    
    # Drift increases the loop gain
    drift = []
    for _ in range(50):
        x = np.random.normal(0, 0.2, 3)
        x[1] += 1.2 * x[0]
        x[2] += 1.2 * x[1]
        x[0] += 1.2 * x[2]
        drift.append(x)
        
    nerve.watch(drift)
    
    # Assert that the system gets stuck and fails to resolve the cycle
    # (since the DAG constraint forces it to break the cycle, but Lyapunov gradient pushes it back)
    health = nerve.structural_health()
    if health.status != "degraded":
        print("WARNING: Documented failure no longer reproduces. Update FAILURES.md and move to Resolved Issues section.")
        assert False, "Failure C no longer reproduces: system escaped local minimum."

if __name__ == "__main__":
    test_lyapunov_local_minima()
    print("Failure C confirmed: System gets trapped in local minimum during cyclic feedback.")
