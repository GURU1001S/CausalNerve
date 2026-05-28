"""
examples/turbofan_demo.py
=========================
Simulates industrial turbofan degradation, structural alarm trigger,
and graph surgery self-repair. Runnable in <1 minute.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causalnerve.api import CausalNerve

def run_turbofan_demo():
 print("=== STARTING TURBOFAN DEGRADATION DEMO ===")
 
 # 1. Initialize CausalNerve
 nerve = CausalNerve(nodes=6, device="cpu")
 nerve.node_labels = {
 0: "Fuel Flow",
 1: "Core Fan Speed (N1)",
 2: "High-Pressure Turbine Speed (N2)",
 3: "Exhaust Gas Temp (EGT)",
 4: "Vibration Sensor",
 5: "Oil Pressure"
 }
 
 # 2. Simulate streaming data
 print("\n[Step 1] Streaming normal engine operation...")
 for cycle in range(50):
 # Normal correlated states
 state = np.random.normal(0, 0.1, size=6)
 state[1] = 0.8 * state[0] + np.random.normal(0, 0.05) # Fuel -> N1
 state[2] = 0.7 * state[1] + np.random.normal(0, 0.05) # N1 -> N2
 state[3] = 0.9 * state[2] + np.random.normal(0, 0.05) # N2 -> EGT
 nerve.step(state)
 
 print(f"Initial Health: {nerve.structural_health().status.upper()}")
 print(f"Initial Lyapunov Energy V(G): {nerve.lyapunov.current_energy:.4f}")
 
 # 3. Inject gradual high-pressure turbine degradation
 print("\n[Step 2] Injecting gradual High-Pressure Turbine thermal degradation...")
 for cycle in range(30):
 state = np.random.normal(0, 0.1, size=6)
 # N2 -> EGT link breaks, EGT leaks energy to Vibration
 state[1] = 0.8 * state[0]
 state[2] = 0.7 * state[1]
 state[3] = 0.3 * state[2] + 0.9 * state[4] + np.random.normal(0, 0.2) # Anomaly
 nerve.step(state)

 # 4. Query Explanation and Causal Backtracking
 print("\n[Step 3] Querying CausalNerve Natural-Language Diagnostics...")
 
 # Query root cause of EGT sensor anomaly
 why_result = nerve.why("Exhaust Gas Temp (EGT)", mode="engineering")
 print(why_result.explanation)
 
 # Query counterfactual what-if intervention
 print("\n[Step 4] Simulating counterfactual throttle intervention...")
 what_if_res = nerve.what_if({"Fuel Flow": -1.5}, mode="engineering")
 print(what_if_res.explanation)
 
 print("\n=== TURBOFAN DEMO COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
 run_turbofan_demo()
