import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causalnerve.api import CausalNerve

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def test_oscillation():
    print("\nRunning Edge Oscillation Stress Test...")
    set_seed()
    nerve = CausalNerve(nodes=21)
    validator = nerve.ocgr.live_validator
    
    # Force rapid oscillation
    print("Simulating rapid graph edits on Edge (4, 9)...")
    for cycle in range(10):
        edge = (4, 9)
        validator.edge_flip_counts[edge] = validator.edge_flip_counts.get(edge, 0) + 1
        is_oscillating = validator.check_oscillation(edge)
        if is_oscillating:
            print(f"[SAFE] Cycle {cycle}: Edge {edge} correctly QUARANTINED by Anti-Oscillation Lock.")
            break
            
    if (4, 9) in validator.metrics.quarantined_edges:
        print("-> Oscillation stress test PASSED.")
    else:
        print("-> [UNSAFE] Oscillation test FAILED.")

def test_rollbacks():
    print("\nRunning Rollback Stress Test...")
    set_seed()
    nerve = CausalNerve(nodes=21)
    validator = nerve.ocgr.live_validator
    
    print("Mocking accepted surgery on Edge (1, 2) at cycle 10...")
    validator.active_surgeries[(1, 2)] = 10
    validator.edge_history[(1, 2)] = [{'cycle': 10, 'confidence': 0.8}]
    
    print("Fast-forwarding to cycle 20. Simulating massive leakage spike...")
    current_leakages = {(1, 2): 0.50} # Massive spike
    rollbacks = validator.monitor_persistence(20, current_leakages)
    
    print(f"Rollbacks triggered: {rollbacks}")
    if (1, 2) in rollbacks:
        print("[SAFE] Rollback successfully activated on persistent failure.")
        print("-> Rollback stress test PASSED.")
    else:
        print("[UNSAFE] Rollback failed to trigger.")

if __name__ == "__main__":
    test_oscillation()
    test_rollbacks()
