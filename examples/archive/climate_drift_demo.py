"""
examples/climate_drift_demo.py
==============================
Simulates long-term regional climate drift, structural shifts,
and sensor correlation tracking. Runnable in <1 minute.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causalnerve.api import CausalNerve

def run_climate_demo():
 print("=== STARTING CLIMATE DRIFT DEMO ===")
 
 # Initialize CausalNerve
 nerve = CausalNerve(nodes=5, device="cpu")
 nerve.node_labels = {
 0: "Sea Surface Temperature (SST)",
 1: "Specific Humidity",
 2: "Zonal Wind Speed",
 3: "Precipitation Rate",
 4: "Land Surface Temperature"
 }
 
 # 1. Base climate regime
 print("\n[Step 1] Streaming historical climate baseline data...")
 for _ in range(40):
 state = np.random.normal(0, 0.05, size=5)
 # SST drive humidity, driving precipitation
 state[1] = 0.65 * state[0]
 state[3] = 0.5 * state[1]
 nerve.step(state)
 
 print(f"Baseline Lyapunov Energy: {nerve.lyapunov.current_energy:.4f}")
 
 # 2. Gradual warm drift
 print("\n[Step 2] Injecting long-term warm SST thermal drift...")
 for _ in range(30):
 state = np.random.normal(0, 0.05, size=5)
 state[0] += 0.4 # Gradual SST warming
 state[1] = 0.85 * state[0] + np.random.normal(0, 0.02)
 state[3] = 0.7 * state[1] + np.random.normal(0, 0.02)
 nerve.step(state)
 
 # 3. Query Diagnostics
 print("\n[Step 3] Querying CausalNerve natural-language diagnosis...")
 why_result = nerve.why("Precipitation Rate", mode="engineering")
 print(why_result.explanation)
 
 # 4. Intervention scenario
 print("\n[Step 4] Querying what_if() cloud-seeding precipitation intervention...")
 what_if_res = nerve.what_if({"Specific Humidity": 1.2}, mode="engineering")
 print(what_if_res.explanation)
 
 print("\n=== CLIMATE DRIFT DEMO COMPLETED ===")

if __name__ == "__main__":
 run_climate_demo()
