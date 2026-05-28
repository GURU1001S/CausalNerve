import sys
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.api import CausalNerve
from causalnerve.runtime.stream import LiveCMAPSSStream
from causalnerve.runtime.scheduler import LiveMonitoringScheduler
from causalnerve.visualization_stub.live_graph import LiveGraphVisualizer

def main():
 print("================================================================================")
 print(" CausalNerve: TRUE LIVE LEARNING DEMO (NASA C-MAPSS)")
 print("================================================================================")
 
 # 1. Initialize CausalNerve and Visualizer
 print("[INIT] Booting CausalNerve Engine (Nodes=21)...")
 nerve = CausalNerve(nodes=21, state_dim=64)
 vis = LiveGraphVisualizer(num_nodes=21)
 
 # 2. Connect NASA C-MAPSS Stream
 print("[STREAM] Connecting to raw NASA C-MAPSS Telemetry (FD001)...")
 stream = LiveCMAPSSStream(engine_id=1, realtime=False)
 scheduler = LiveMonitoringScheduler(nerve, stream)
 
 print("-" * 80)
 print("Monitoring Live Feed and Learning the Causal Structure...")
 print("This will run for 500 cycles to observe natural graph evolution.")
 print("Please wait while it processes...")
 
 def on_cycle(cycle, data, state):
 # We will let the adaptation engine (OCGR) naturally learn the graph!
 # No fake anomalies injected.
 
 # Every 10 cycles, log progress
 if cycle % 10 == 0:
 adj = nerve.graph.get_dense_adjacency()
 active_edges = (adj > 0.01).sum().item()
 print(f"[Cycle {cycle:03d}] Active Edges: {active_edges} | V(G) Energy: {state.lyapunov_history[-1]:.3f} | Alarms: {len(state.active_alarms)}")
 
 adj_cpu = nerve.graph.get_dense_adjacency().cpu().numpy()
 leak = state.leakage_history[-1] if state.leakage_history else 0.0
 energy = state.lyapunov_history[-1] if state.lyapunov_history else 0.0
 
 recent_alarms = [a for a in state.active_alarms if cycle - a['cycle'] < 3]
 recent_surgeries = [s for s in state.accepted_surgeries if cycle - s.get('cycle', -1) < 3]
 
 # Record the visual state
 vis.update(cycle, adj_cpu, leak, energy, recent_alarms, recent_surgeries)
 
 # Run for a longer horizon to observe learning
 if cycle >= 500:
 raise InterruptedError("Learning Complete")

 try:
 scheduler.run(on_cycle=on_cycle)
 except InterruptedError:
 pass
 
 print("-" * 80)
 print("[SUCCESS] Live stream monitoring complete.")
 
 html_output = "nasa_true_learning_evolution.html"
 vis.render_animation_html(html_output)
 
 print("\nVisualizer Dashboard Generated:")
 print(f" -> Open '{os.path.abspath(html_output)}' in your web browser!")
 print(" Click 'Play Stream' at the bottom to watch the graph structurally adapt over time.")
 print("================================================================================")

if __name__ == "__main__":
 main()
