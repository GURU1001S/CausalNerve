import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import threading
import numpy as np
from causalnerve.api import CausalNerve
from causalnerve.datasets.eeg_real import RealEEGDataset
from causalnerve.plugins.registry import PluginRegistry
from causalnerve.plugins.eeg_plugin import EEGDomainPlugin
from causalnerve_observatory import CausalNerveObservatory

# Register EEG domain explicitly since it's not in a sub-module
PluginRegistry.register(EEGDomainPlugin())

def detect_events(result_state, history_leakages):
 """EEG Event detection heuristics for printing."""
 events = []
 current_leak = result_state.leakage
 
 # 1. Synchronization Burst (sudden drop in leakages across the board)
 if len(history_leakages) > 5:
 if current_leak < 0.5 * np.mean(history_leakages[-5:]):
 events.append("SYNCHRONIZATION BURST")
 
 # 2. Seizure-like Connectivity Explosion (many alarms firing together)
 if len(result_state.alarms) > 3:
 events.append("SEIZURE-LIKE TOPOLOGY EXPLOSION")
 
 # 3. Sudden Topology Rewiring
 if result_state.graph_changed:
 events.append("SUDDEN REWIRING")
 
 return events

def run_live_connectivity():
 print("="*60)
 print(" CAUSALNERVE — EEG LIVE CONNECTIVITY DEMO")
 print("="*60)
 
 # 1. Initialize API for EEG domain
 nerve = CausalNerve.from_preset("eeg")
 print(f"[*] Initialized CausalNerve for Domain: eeg")
 print(f"[*] Nodes tracked: {list(nerve._node_labels.values())}")
 
 # 2. Start the Observatory Dashboard in EEG mode
 obs = CausalNerveObservatory(port=8765, scenario='eeg', auto_open=True)
 obs.start()
 print("[*] Dashboard UI started on http://localhost:8765")
 
 # 3. Load Real EEG Stream
 dataset = RealEEGDataset(subject=1, run=4)
 print(f"[*] Connecting to MNE EEG Stream... (Subject 1, Run 4)")
 try:
 stream = dataset.stream_subject(window_size=128, step=64)
 except Exception as e:
 print(f"[!] Could not load real EEG dataset: {e}. Generating synthetic stream...")
 def synth_stream():
 for i in range(100):
 yield np.random.randn(128, len(nerve.nodes))
 stream = synth_stream()

 print("[*] Beginning live monitoring. Press Ctrl+C to stop.")
 time.sleep(1)
 
 history_leakages = []
 try:
 for i, window in enumerate(stream):
 # We treat the window's instantaneous correlation or mean absolute activity 
 # as the telemetry snapshot for the frame, or just run a frame by frame step.
 # In this demo, we simulate frame-by-frame telemetry from the window.
 # We'll take the mean absolute amplitude of the window as the state proxy.
 telemetry = np.mean(np.abs(window), axis=0)
 
 # Artificial anomaly injection for demo: Motor imagery transition
 if i == 15:
 print("\n[!] >> INJECTING VIRTUAL MOTOR IMAGERY TRANSITION <<")
 # Boost C3, C4, Cz (indices 8, 9, 10 in our canonical array)
 telemetry[8:11] *= 3.0
 
 if i == 30:
 print("\n[!] >> VIRTUAL CORTICAL SUPPRESSION (Intervention) <<")
 print(f" Executing: do(Cz = 0.1)")
 do_result = nerve.do("Cz", 0.1)
 print(f" Isolation verified: {do_result['isolation_verified']}")
 print(f" Affected Descendants: {do_result['descendants_affected']}")
 # Compare factual vs intervened
 whatif = nerve.what_if("Cz", 0.1)
 print(f" Counterfactual divergence: {whatif['cumulative_divergence']:.3f}")
 print(f" Leakage Reduction: {whatif['leakage_reduction']:.3f}\n")
 
 # Watch cycle
 result = nerve.watch(telemetry)
 history_leakages.append(result.leakage)
 
 # Send live state to the Observatory UI
 # Convert edge probabilities from adjacency
 adj = nerve.graph.adj.copy()
 # Normalize to 0-1 for probabilities
 max_w = np.max(np.abs(adj)) if np.max(np.abs(adj)) > 0 else 1.0
 edge_probs = (np.abs(adj) / max_w).tolist()
 
 obs.update(result.cycle, {
 'sensor_values': telemetry.tolist(),
 'leakage_L': result.leakage,
 'lyapunov_V': result.leakage * 10, # proxy
 'divergence': 0.0,
 'edge_probs': edge_probs,
 'world0_states': telemetry.tolist(),
 'world1_states': telemetry.tolist(),
 'alarm_nodes': [],
 'ocgr_events': result.revisions
 })
 
 # Event Detection
 events = detect_events(result, history_leakages)
 
 # Reporting
 print(f"Cycle {result.cycle:03d} | Leakage L(G): {result.leakage:.4f} | "
 f"Alarms: {len(result.alarms)} | "
 f"Events: {', '.join(events) if events else 'Stable'}")
 
 for rev in result.revisions:
 print(f" [!] Graph Surgery: {rev['rationale']}")
 
 time.sleep(0.1)
 
 if i > 50:
 break
 
 except KeyboardInterrupt:
 print("Stopped.")

if __name__ == "__main__":
 run_live_connectivity()
