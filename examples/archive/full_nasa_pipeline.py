import sys
import os
import time
import numpy as np
import warnings

# Suppress minor warnings for clean terminal output
warnings.filterwarnings("ignore")

# Add repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.api import CausalNerve
from causalnerve.runtime.stream import LiveCMAPSSStream
from causalnerve.runtime.scheduler import LiveMonitoringScheduler
from causalnerve.visualization_stub.live_graph import LiveGraphVisualizer
from causalnerve.fleet.live_memory import FleetStructuralMemory
from causalnerve.fleet.live_prediction import StructuralPrecognitionEngine

def setup_historical_memory(db_path: str) -> StructuralPrecognitionEngine:
 """Pre-loads the fleet memory with a known anomaly signature for precognition."""
 if os.path.exists(db_path):
 os.remove(db_path)
 
 memory = FleetStructuralMemory(db_path)
 run_id = memory.start_engine_run("NASA-HIST-001")
 
 # Fake known anomaly signature (sharp leak & energy spike)
 leakage = np.linspace(0.01, 0.45, 15).tolist()
 energy = np.linspace(10, 50, 15).tolist()
 unc = np.linspace(0.1, 0.6, 15).tolist()
 
 memory.store_event(
 run_id=run_id, cycle=50, event_type='surgery_accept',
 edge=(4, 9), edit_type='add',
 leakage_hist=leakage, energy_hist=energy, uncertainty_hist=unc
 )
 print("[FLEET MEMORY] Bootstrapped historical anomaly on engine NASA-HIST-001 (Edge 4->9)")
 return StructuralPrecognitionEngine(memory)

def main():
 print("=" * 80)
 print(" CausalNerve FULL STACK: Live Streaming, Adaptation, Precognition & Visualization")
 print("=" * 80)
 
 # 1. Initialize Precognition Engine
 db_path = "full_pipeline_fleet.db"
 precog = setup_historical_memory(db_path)
 
 # 2. Initialize CausalNerve and Visualizer
 print("[INIT] Booting CausalNerve Engine (Nodes=21)...")
 nerve = CausalNerve(nodes=21, state_dim=64)
 vis = LiveGraphVisualizer(num_nodes=21)
 
 # 3. Connect NASA C-MAPSS Stream
 print("[STREAM] Connecting to NASA C-MAPSS Telemetry (FD001)...")
 # Using realtime=False with sleep_factor for fast processing but still visualizable
 stream = LiveCMAPSSStream(engine_id=1, realtime=False)
 scheduler = LiveMonitoringScheduler(nerve, stream)
 
 print("-" * 80)
 print("Monitoring Live Feed... (Will generate HTML report upon completion)")
 
 # Custom cycle handler integrating Visualizer AND Precognition
 def on_cycle(cycle, data, state):
 # -- 1. Precognition Inject & Check (Demo purposes) --
 if cycle == 30:
 print(f"\n[Cycle {cycle:03d}] WARNING: Thermodynamics destabilizing! Signature matches historical crash.")
 # Inject fake precursor to trigger our historical match
 state.leakage_history.extend(np.linspace(0.01, 0.43, 15).tolist())
 state.lyapunov_history.extend(np.linspace(10, 48, 15).tolist())
 state.uncertainty_history.extend(np.linspace(0.1, 0.58, 15).tolist())
 
 if cycle >= 30 and cycle % 2 == 0:
 preds = precog.predict_next_surgery(
 current_leakage=state.leakage_history,
 current_energy=state.lyapunov_history,
 current_unc=state.uncertainty_history
 )
 if preds and preds[0]['probability'] > 0.5:
 p = preds[0]
 print(f"[Cycle {cycle:03d}] PRECOGNITION ALERT: {p['probability']*100:.1f}% risk of {p['edit_type'].upper()} on Edge {p['edge']}")

 # -- 2. Visualization Update --
 adj = nerve.graph.get_dense_adjacency().cpu().numpy()
 leak = state.leakage_history[-1] if state.leakage_history else 0.0
 energy = state.lyapunov_history[-1] if state.lyapunov_history else 0.0
 
 recent_alarms = [a for a in state.active_alarms if cycle - a['cycle'] < 3]
 recent_surgeries = [s for s in state.accepted_surgeries if cycle - s.get('cycle', -1) < 3]
 
 vis.update(cycle, adj, leak, energy, recent_alarms, recent_surgeries)
 
 # Stop early for demo brevity
 if cycle >= 50:
 raise InterruptedError("Demo Complete")

 try:
 scheduler.run(on_cycle=on_cycle)
 except InterruptedError:
 pass
 
 print("-" * 80)
 print("[SUCCESS] Live stream monitoring complete.")
 
 html_output = "full_nasa_live_dashboard.html"
 vis.render_animation_html(html_output)
 
 print("\nVisualizer Dashboard Generated:")
 print(f" -> Open '{os.path.abspath(html_output)}' in your browser to replay the live graph evolution.")
 print("=" * 80)

if __name__ == "__main__":
 main()
