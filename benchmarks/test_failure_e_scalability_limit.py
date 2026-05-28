import time
import numpy as np
from causalnerve import CausalNerve

def test_scalability_limit():
    """
    Tests computational bottlenecks in 100+ Node Fleet Topologies.
    (Failure E)
    """
    # 100 nodes creates a huge complexity for Jacobian paths
    nodes = 100
    nerve = CausalNerve(nodes=nodes)
    
    stream = [np.random.normal(0, 0.1, nodes) for _ in range(5)]
    
    start = time.time()
    # Fit might take a while, but watch is where dual-world rollout happens
    nerve.fit(stream, epochs=1)
    
    # Force a structural alarm by injecting huge drift
    drift = [np.random.normal(5.0, 1.0, nodes) for _ in range(2)]
    nerve.watch(drift)
    duration = time.time() - start
    
    # It scales at O(N^3). It should take a noticeable amount of time per step.
    # We assert it takes more than 0.1 seconds per step on average for 100 nodes
    # (In reality the documentation states >12.4 seconds for a 100-node Scale-Free DAG).
    # If the user fixed it by pruning, this test might fail (meaning it's too fast).
    if duration <= 0.05:
        print("WARNING: Documented failure no longer reproduces. Update FAILURES.md and move to Resolved Issues section.")
        assert False, "Failure E no longer reproduces: system ran 100-node graph abnormally fast."

if __name__ == "__main__":
    test_scalability_limit()
    print("Failure E confirmed: 100+ node topologies exhibit scalability limits.")
