"""
stress_test_alarm_scaling.py
============================
Evaluates the robustness of StructuralAlarmSystemV2 under catastrophic cascading failure
on large-scale graphs (N=500+). Proves that alarm storms are localized and suppressed.
"""

import numpy as np
import time
from causalnerve.adaptation.alarm_system_v2 import StructuralAlarmSystemV2

class MockGraphEngine:
    def __init__(self, n: int):
        self.n = n
        # Sparse Erdos-Renyi
        self.adj = (np.random.rand(n, n) < 0.05).astype(float)
        
    def get_dense_adjacency(self):
        return self.adj

def run_stress_test():
    print("--- CausalNerve Alarm System Scalability Stress Test ---")
    N_NODES = 500
    N_CYCLES = 100
    
    engine = MockGraphEngine(N_NODES)
    alarm_system = StructuralAlarmSystemV2(threshold_on=0.08, threshold_off=0.04, persistence=3)
    
    # Simulate a cascade: Node 42 fails catastrophically at cycle 20.
    # Its failure causes all 25 of its outgoing edges to leak massively.
    # Without cascade suppression, this would trigger 25+ simultaneous alarms (a storm).
    
    root_node = 42
    downstream_nodes = [v for v in range(N_NODES) if engine.adj[root_node, v] > 0]
    print(f"Node {root_node} has {len(downstream_nodes)} downstream connections.")
    
    total_alarms_fired = 0
    start_time = time.time()
    
    for cycle in range(N_CYCLES):
        leakage = np.zeros((N_NODES, N_NODES))
        
        # Inject standard noise
        leakage += np.random.normal(0.01, 0.02, (N_NODES, N_NODES))
        
        if cycle >= 20:
            # The cascade begins
            for v in downstream_nodes:
                leakage[root_node, v] = 0.50 # Massive leak
                # Derivative leaks: the downstream nodes also start leaking to their children
                children = [k for k in range(N_NODES) if engine.adj[v, k] > 0]
                for k in children:
                    leakage[v, k] = 0.30
                    
        events = alarm_system.process_leakage_matrix(leakage, cycle, engine)
        
        if events:
            total_alarms_fired += len(events)
            print(f"Cycle {cycle}: Fired {len(events)} prioritized alarms. Top target: Edge {events[0].edge}")
            
    elapsed = time.time() - start_time
    print(f"\\n--- Test Complete ---")
    print(f"Runtime: {elapsed:.3f} seconds")
    print(f"Total Alarms Emitted: {total_alarms_fired}")
    
    # Assertions for robustness
    assert total_alarms_fired <= 60, f"Storm detected! System fired {total_alarms_fired} alarms."
    print("SUCCESS: Alarm cascade successfully suppressed and localized.")

if __name__ == "__main__":
    run_stress_test()
