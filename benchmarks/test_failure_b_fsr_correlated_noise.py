import numpy as np
from causalnerve import CausalNerve

def test_fsr_correlated_noise():
    """
    Tests False Surgery Rate under highly correlated noise.
    (Failure B)
    """
    nerve = CausalNerve(nodes=4)
    
    # Create perfectly correlated noise
    stream = []
    for _ in range(100):
        # Node 0 and 1 are perfectly correlated
        noise = np.random.normal(0, 0.1)
        x = np.zeros(4)
        x[0] = noise
        x[1] = noise
        x[2] = 0.5 * noise # Effect
        x[3] = np.random.normal(0, 0.1)
        stream.append(x)
        
    nerve.fit(stream[:50], epochs=5)
    
    # When drifting, the correlation remains but amplitude increases
    drift = []
    for _ in range(50):
        noise = np.random.normal(0, 0.5)
        x = np.zeros(4)
        x[0] = noise
        x[1] = noise
        x[2] = 0.5 * noise
        x[3] = np.random.normal(0, 0.1)
        drift.append(x)
        
    res = nerve.watch(drift)
    
    # Under regime ambiguity, false surgery rate is high. 
    # We assert that the graph proposes or accepts multiple edits due to the ambiguity.
    # We expect some edits to have been proposed due to ambiguity
    failure_occurred = (len(res.revisions) > 0 or res.graph_changed)
    if not failure_occurred:
        print("WARNING: Documented failure no longer reproduces. Update FAILURES.md and move to Resolved Issues section.")
        assert False, "Failure B no longer reproduces: system did not make false surgeries under correlated noise."

if __name__ == "__main__":
    test_fsr_correlated_noise()
    print("Failure B confirmed: Correlated noise causes false surgery proposals.")
