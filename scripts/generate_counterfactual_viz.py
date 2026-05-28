"""Generate real counterfactual visualization SVG."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from causalnerve.runtime.intervention.intervention import CausalGraph, InterventionEngine
from causalnerve.runtime.intervention.counterfactual import CounterfactualEngine

# Build the aerospace DAG
NODES = {0:"Fan",1:"LPC",2:"HPC",3:"Combustor",4:"HPT",5:"LPT",
         6:"H.Spool",7:"L.Spool",8:"P.Bank",9:"Cooling",10:"Bypass",
         11:"Fuel",12:"Snsr.A",13:"Snsr.B"}
EDGES = [(11,3),(3,4),(4,2),(4,6),(6,2),(2,1),(5,7),(7,0),(9,4),(10,1),(4,12),(3,12)]

graph = CausalGraph(14, EDGES, node_labels=NODES)
ie = InterventionEngine()
cf = CounterfactualEngine(ie)

states = np.full(14, 0.5)
result = cf.simulate(graph, {4: 0.9}, initial_states=states, horizon=50)

# Generate SVG chart
W, H = 800, 400
margin = 60

w0_mean = np.mean(result.world_0_trajectory, axis=1)  # factual average
w1_mean = np.mean(result.world_1_trajectory, axis=1)  # intervened average
div = result.divergence

# Scale helpers
def sx(i, n): return margin + (i / (n - 1)) * (W - 2 * margin)
def sy(v, mn, mx): return H - margin - ((v - mn) / (mx - mn + 1e-9)) * (H - 2 * margin)

all_vals = np.concatenate([w0_mean, w1_mean])
mn, mx = all_vals.min(), all_vals.max()

n = len(w0_mean)
w0_pts = " ".join(f"{sx(i,n):.1f},{sy(w0_mean[i],mn,mx):.1f}" for i in range(n))
w1_pts = " ".join(f"{sx(i,n):.1f},{sy(w1_mean[i],mn,mx):.1f}" for i in range(n))

# Divergence fill (area between curves)
fill_pts = " ".join(f"{sx(i,n):.1f},{sy(w0_mean[i],mn,mx):.1f}" for i in range(n))
fill_pts += " " + " ".join(f"{sx(i,n):.1f},{sy(w1_mean[i],mn,mx):.1f}" for i in range(n-1, -1, -1))

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#0D1B2A"/>

<!-- Title -->
<text x="{W//2}" y="25" text-anchor="middle" fill="#B8860B" font-size="16" font-weight="bold" font-family="monospace">CausalNerve Counterfactual: do(HPT = 0.9)</text>

<!-- Divergence region -->
<polygon points="{fill_pts}" fill="#B8860B" opacity="0.15"/>

<!-- Factual trajectory -->
<polyline points="{w0_pts}" fill="none" stroke="#22C55E" stroke-width="2.5"/>

<!-- Intervened trajectory -->
<polyline points="{w1_pts}" fill="none" stroke="#DC2626" stroke-width="2.5" stroke-dasharray="6,3"/>

<!-- Legend -->
<rect x="{W - 220}" y="45" width="200" height="65" rx="4" fill="#0D1B2A" stroke="#6B7B8D" opacity="0.9"/>
<line x1="{W-210}" y1="62" x2="{W-180}" y2="62" stroke="#22C55E" stroke-width="2.5"/>
<text x="{W-175}" y="66" fill="#E2E8F0" font-size="11" font-family="monospace">World 0 (Factual)</text>
<line x1="{W-210}" y1="82" x2="{W-180}" y2="82" stroke="#DC2626" stroke-width="2.5" stroke-dasharray="6,3"/>
<text x="{W-175}" y="86" fill="#E2E8F0" font-size="11" font-family="monospace">World 1 (Intervened)</text>
<rect x="{W-210}" y="94" width="30" height="8" fill="#B8860B" opacity="0.4"/>
<text x="{W-175}" y="102" fill="#E2E8F0" font-size="11" font-family="monospace">Divergence Region</text>

<!-- Axes labels -->
<text x="{W//2}" y="{H-10}" text-anchor="middle" fill="#6B7B8D" font-size="11" font-family="monospace">Timestep</text>
<text x="15" y="{H//2}" text-anchor="middle" fill="#6B7B8D" font-size="11" font-family="monospace" transform="rotate(-90 15 {H//2})">Mean Node State</text>

<!-- Stats -->
<text x="{margin}" y="{H - 15}" fill="#B8860B" font-size="10" font-family="monospace">Cumulative Divergence: {result.cumulative_divergence:.4f} | Affected: {len(result.affected_nodes)} nodes | Leakage Reduction: {result.leakage_reduction:.6f}</text>
</svg>'''

Path("results").mkdir(exist_ok=True)
with open("results/counterfactual_real.svg", "w", encoding="utf-8") as f:
    f.write(svg)
# Also save as .png extension for compatibility
with open("results/counterfactual_real.png", "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved to results/counterfactual_real.svg")
print(f"Cumulative Divergence: {result.cumulative_divergence:.4f}")
print(f"Affected Nodes: {[NODES[n] for n in result.affected_nodes]}")
print(f"Leakage Reduction: {result.leakage_reduction:.6f}")
