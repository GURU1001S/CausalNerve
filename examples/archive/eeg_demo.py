"""
examples/eeg_demo.py
====================
Simulates neural connectivity shifts (visual stimulus excitation),
causal backtracking, and brain state adaptation. Runnable in <1 minute.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causalnerve.api import CausalNerve

def run_eeg_demo():
 print("=== STARTING BRAIN CONNECTIVITY EEG DEMO ===")
 
 # Initialize CausalNerve
 nerve = CausalNerve(nodes=5, device="cpu")
 nerve.node_labels = {
 0: "Occipital Channel (O1)",
 1: "Parietal Channel (P1)",
 2: "Temporal Channel (T3)",
 3: "Frontal Channel (F1)",
 4: "Motor Strip (C3)"
 }
 
 # 1. Base Rest State EEG
 print("\n[Step 1] Recording resting-state EEG signals...")
 for _ in range(40):
 state = np.random.normal(0, 0.05, size=5)
 # Sleepy background rhythm
 state[1] = 0.5 * state[0]
 state[3] = 0.3 * state[1]
 nerve.step(state)
 
 print(f"Resting Lyapunov Energy V(G): {nerve.lyapunov.current_energy:.4f}")
 
 # 2. Stimulus Excitation
 print("\n[Step 2] Applying visual stimulus (flash) to Occipital node...")
 for _ in range(30):
 state = np.random.normal(0, 0.05, size=5)
 # Flash activates Occipital -> Parietal -> Frontal cascade
 state[0] += 0.8 # Occipital stimulus spike
 state[1] = 0.85 * state[0] + np.random.normal(0, 0.02)
 state[3] = 0.75 * state[1] + np.random.normal(0, 0.02)
 nerve.step(state)
 
 # 3. Query academic explanations
 print("\n[Step 3] Querying Academic-Grade Causal Explanations...")
 why_result = nerve.why("Frontal Channel (F1)", mode="academic")
 print(why_result.explanation)
 
 # 4. Query counterfactual motor inhibition do()
 print("\n[Step 4] Querying what_if() motor strip suppression...")
 what_if_res = nerve.what_if({"Motor Strip (C3)": -0.8}, mode="academic")
 print(what_if_res.explanation)
 
 print("\n=== EEG BRAIN STIMULUS DEMO COMPLETED ===")

if __name__ == "__main__":
 run_eeg_demo()
