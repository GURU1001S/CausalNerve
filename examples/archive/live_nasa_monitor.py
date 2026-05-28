import sys
import os

# Add repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.api import CausalNerve

def main():
 print("=" * 70)
 print(" CausalNerve - Phase 1: Live NASA C-MAPSS Monitoring Stack")
 print("=" * 70)
 
 print("\n[INIT] Initializing CausalNerve...")
 # 21 sensors in CMAPSS dataset
 nerve = CausalNerve(nodes=21, state_dim=64)
 nerve.alarm_threshold = 0.05
 
 print("[INIT] Starting live stream on Engine 1 (FD001)...")
 print("-" * 70)
 
 def on_cycle(cycle, data, state):
 if cycle % 10 == 0 or cycle == 0:
 rul = data['rul']
 edges = state.current_edges
 leakage = state.leakage_history[-1] if state.leakage_history else 0.0
 print(f"Cycle {cycle:03d} | RUL: {rul:03.0f} | Active Edges: {edges:02d} | Causal Leakage: {leakage:.4f}")
 
 if len(state.accepted_surgeries) > 0 and cycle == state.accepted_surgeries[-1].get("cycle"):
 surg = state.accepted_surgeries[-1]
 u, v = surg['edge']
 edit_type = surg['edit_type']
 print(f"\n>>> [ALARM] Structural Alarm Triggered at cycle {cycle}!")
 print(f">>> [SURGERY] Accepted {edit_type.upper()} on structural dependency ({u} -> {v})")
 
 # Print causal explanation
 print(f">>> [EXPLANATION] Network adapted to causal leakage shift involving sensor {u}")
 print("-" * 70)

 # We use a sleep_factor to simulate streaming realtime
 state = nerve.live_watch(engine_id=1, realtime=True, sleep_factor=0.01, loop=False, on_cycle=on_cycle)
 
 print("=" * 70)
 print("Stream finished.")
 print(f"Total Alarms Fired: {len(state.active_alarms)}")
 print(f"Total Surgeries Accepted: {len(state.accepted_surgeries)}")
 print(f"Final Graph Edges: {state.current_edges}")
 print("=" * 70)

if __name__ == "__main__":
 main()
