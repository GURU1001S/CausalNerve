import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.fleet.live_memory import FleetStructuralMemory
from causalnerve.fleet.live_prediction import StructuralPrecognitionEngine
from causalnerve.api import CausalNerve
from causalnerve.runtime.stream import LiveCMAPSSStream
from causalnerve.runtime.scheduler import LiveMonitoringScheduler

def mock_fleet_history(memory: FleetStructuralMemory):
 # Mocking an engine that suffered a structural shift at cycle 50
 run_id = memory.start_engine_run("NASA-HIST-001")
 
 # We create a fake 15-cycle thermodynamic precursor signature 
 # where leakage and energy spike
 leakage = np.linspace(0.01, 0.45, 15).tolist()
 energy = np.linspace(10, 50, 15).tolist()
 unc = np.linspace(0.1, 0.6, 15).tolist()
 
 memory.store_event(
 run_id=run_id,
 cycle=50,
 event_type='surgery_accept',
 edge=(4, 9),
 edit_type='add',
 leakage_hist=leakage,
 energy_hist=energy,
 uncertainty_hist=unc
 )
 print("[FLEET DB] Bootstrapped historical NASA structural anomaly on engine NASA-HIST-001 (Edge 4->9)")

def main():
 print("=" * 70)
 print(" CausalNerve - Phase 4: Structural Precognition (Fleet Memory)")
 print("=" * 70)
 
 db_path = "fleet_live_memory.db"
 if os.path.exists(db_path):
 os.remove(db_path) # Clean slate for demo
 
 memory = FleetStructuralMemory(db_path)
 mock_fleet_history(memory)
 
 precog = StructuralPrecognitionEngine(memory)
 
 print("\n[STREAM] Starting live prediction monitoring on Engine LIVE-002...")
 nerve = CausalNerve(nodes=21, state_dim=64)
 stream = LiveCMAPSSStream(engine_id=1, realtime=False)
 scheduler = LiveMonitoringScheduler(nerve, stream)
 
 # To trigger our DTW, we need to inject a similar precursor in the live state
 
 def on_cycle(cycle, data, state):
 if cycle == 30:
 print("\n[WARNING] Thermodynamics destabilizing! Injecting synthetic precursor shift...")
 state.leakage_history.extend(np.linspace(0.01, 0.43, 15).tolist())
 state.lyapunov_history.extend(np.linspace(10, 48, 15).tolist())
 state.uncertainty_history.extend(np.linspace(0.1, 0.58, 15).tolist())
 
 if cycle >= 30 and cycle % 2 == 0:
 preds = precog.predict_next_surgery(
 current_leakage=state.leakage_history,
 current_energy=state.lyapunov_history,
 current_unc=state.uncertainty_history
 )
 
 if preds:
 p = preds[0]
 if p['probability'] > 0.5:
 print(f"\nCycle {cycle:03d} | [PRECOGNITION ALERT]")
 print(f" -> Likely Next Surgery : {p['edit_type'].upper()} Edge {p['edge']}")
 print(f" -> Probability : {p['probability']*100:.1f}%")
 print(f" -> Precursor Sim Score : {p['precursor_similarity']:.3f}")
 print(f" -> Matched Engines : {p['matched_historical_engines']}")
 print("-" * 50)
 
 if cycle > 36:
 sys.exit(0)
 
 try:
 scheduler.run(on_cycle=on_cycle)
 except SystemExit:
 print("\n[SUCCESS] Precognition demo completed successfully.")

if __name__ == "__main__":
 main()
