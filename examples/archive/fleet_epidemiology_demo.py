import sys
import os
import warnings
import numpy as np
import random
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.visualization_stub.fleet_dashboard import FleetEpidemiologyVisualizer

def main():
 print("================================================================================")
 print(" CausalNerve: NASA C-MAPSS FLEET EPIDEMIOLOGY INTELLIGENCE")
 print("================================================================================")
 
 num_engines = 25
 print(f"[INIT] Booting Fleet Simulator (Tracking {num_engines} engines)...")
 vis = FleetEpidemiologyVisualizer(num_engines=num_engines)
 
 # We will simulate 350 cycles
 max_cycles = 350
 
 # Simulate engine degradation parameters
 # Each engine has a 'failure_cycle' where it crosses threshold (0.7)
 failure_cycles = [random.randint(150, 330) for _ in range(num_engines)]
 
 # Motifs library
 motifs = {
 "Fuel Flow -> Combustor Temp (T30)": 0,
 "Combustor Temp -> HPC Pressure (P30)": 0,
 "HPC Temp -> Exhaust Temp (T50)": 0,
 "Fan Speed (Nf) -> BPR": 0
 }
 
 # Assign a primary motif to each engine to simulate recurrence mining
 engine_motifs = [random.choices(list(motifs.keys()), weights=[0.45, 0.35, 0.15, 0.05])[0] for _ in range(num_engines)]
 
 print("-" * 80)
 print("Monitoring Live Fleet Stream and Mining Root Causes...")
 print("-" * 80)
 
 for cycle in range(max_cycles):
 engine_healths = []
 risk_table = []
 
 # Calculate active motifs for this cycle to build the bar chart cumulatively
 current_motifs = {k: 0 for k in motifs}
 
 for i in range(num_engines):
 fc = failure_cycles[i]
 
 # Non-linear degradation curve (logistic-like)
 progress = cycle / fc
 # Health = 0.0 at cycle 0, 0.7 at cycle fc, approaches 1.0 after fc
 if progress < 0.5:
 health = 0.1 * (progress / 0.5)
 elif progress < 1.0:
 health = 0.1 + 0.6 * ((progress - 0.5) / 0.5)
 else:
 health = 0.7 + 0.3 * (min((progress - 1.0) * 2, 1.0))
 
 engine_healths.append(health)
 
 # If engine is actively degrading (health > 0.3), add its motif to counts
 if health > 0.3:
 current_motifs[engine_motifs[i]] += 1
 
 # If engine is at critical risk (RUL < 50 cycles but hasn't failed yet)
 rul = fc - cycle
 if 0 < rul <= 50:
 risk_table.append((f"Engine-{i+1}", f"{rul} cycles", engine_motifs[i]))
 
 # Sort risk table by urgency (RUL)
 risk_table.sort(key=lambda x: int(x[1].split(' ')[0]))
 
 # Keep top 10 most critical for the table to avoid overcrowding
 risk_table = risk_table[:10]
 
 if cycle % 25 == 0:
 crit_count = len(risk_table)
 print(f"[Cycle {cycle:04d}] Engines > 0.7: {sum([1 for h in engine_healths if h > 0.7])} | At Risk (<50c): {crit_count} engines")
 
 vis.update(
 cycle=cycle,
 engine_healths=engine_healths,
 motifs=current_motifs,
 risk_table=risk_table
 )

 print("-" * 80)
 print(f"[SUCCESS] Fleet Epidemiology tracking complete.")
 
 html_output = "nasa_fleet_epidemiology.html"
 vis.render_animation_html(html_output)
 print(f" -> Fleet Dashboard saved to {os.path.abspath(html_output)}")

if __name__ == "__main__":
 main()
