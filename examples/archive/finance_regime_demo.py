"""
examples/finance_regime_demo.py
===============================
Simulates stock market macro regime changes (interest rate hikes),
alarm propagation, and structural tracking. Runnable in <1 minute.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causalnerve.api import CausalNerve

def run_finance_demo():
 print("=== STARTING MARKET REGIME DEMO ===")
 
 # Initialize CausalNerve
 nerve = CausalNerve(nodes=5, device="cpu")
 nerve.node_labels = {
 0: "10-Year Treasury Yields",
 1: "Tech Stocks (QQQ)",
 2: "Energy Stocks (XLE)",
 3: "Crude Oil Price",
 4: "Consumer Discretionary"
 }
 
 # 1. Normal low-rate environment
 print("\n[Step 1] Streaming normal low-interest rate environment data...")
 for _ in range(40):
 state = np.random.normal(0, 0.02, size=5)
 # Tech indices heavily linked to yields (negative)
 state[1] = -0.5 * state[0]
 # Consumer stocks follow tech
 state[4] = 0.6 * state[1]
 nerve.step(state)
 
 print(f"Base Lyapunov Energy: {nerve.lyapunov.current_energy:.4f}")
 
 # 2. Rate hike shock
 print("\n[Step 2] Injecting macro interest rate hike shock...")
 for _ in range(30):
 state = np.random.normal(0, 0.02, size=5)
 # Yields spike, completely breaking tech indices and forcing consumer downturn
 state[0] += 0.5
 state[1] = -1.4 * state[0] + np.random.normal(0, 0.08)
 state[4] = 0.9 * state[1] + np.random.normal(0, 0.08)
 nerve.step(state)
 
 # 3. Query Diagnostics
 print("\n[Step 3] Querying CausalNerve natural-language diagnosis...")
 why_result = nerve.why("Consumer Discretionary", mode="engineering")
 print(why_result.explanation)
 
 # 4. Hedging scenario
 print("\n[Step 4] Querying what_if() hedge proposal on 10-Yr Yields...")
 what_if_res = nerve.what_if({"10-Year Treasury Yields": -0.5}, mode="engineering")
 print(what_if_res.explanation)
 
 print("\n=== FINANCE REGIME DEMO COMPLETED ===")

if __name__ == "__main__":
 run_finance_demo()
