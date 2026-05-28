"""
examples/live_self_repair_demo.py
=================================
A scripted, auto-playing cinematic simulation showcasing CausalNerve's
self-repair, alarms, Lyapunov stability gate, and counterfactual split screen.
"""

import sys
import os
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# Add the project path to PYTHONPATH so it can find causalnerve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causalnerve.visualization_stub.graph_viz import plot_living_graph, THEME
from causalnerve.visualization_stub.counterfactual_viz import plot_counterfactual_split

# Setup a clean 6-node network for the story
N_NODES = 6
node_labels = {
 0: "Fuel Flow",
 1: "Core Speed",
 2: "Exhaust Temp",
 3: "Oil Pressure",
 4: "Vibration Sensor",
 5: "Actuator Feedback"
}

# Base Adjacency Matrix
adj_healthy = np.zeros((N_NODES, N_NODES))
adj_healthy[0, 1] = 0.8
adj_healthy[1, 2] = 0.7
adj_healthy[1, 3] = 0.6
adj_healthy[0, 4] = 0.5
adj_healthy[4, 5] = 0.7

# Repaired Adjacency Matrix
adj_repaired = adj_healthy.copy()
adj_repaired[4, 5] = 0.0
adj_repaired[1, 5] = 0.75

# Define the scenario steps
SCENARIO = [
 {
 "step": 0,
 "name": "Healthy Graph",
 "description": "System is operating normally. Causal paths are stable, leakage is near zero.",
 "alarms": [],
 "node_states": {i: "healthy" for i in range(N_NODES)},
 "adj": adj_healthy,
 "new_edges": [],
 "severed_edges": [],
 "proposed_edges": [],
 "show_split": False,
 "shockwave_node": None,
 "shockwave_progress": 0.0,
 "v_energy": 12.5
 },
 {
 "step": 1,
 "name": "Precursor Warning",
 "description": "Vibration Sensor begins registering micro-fluctuations. Alarm threshold approaching.",
 "alarms": [],
 "node_states": {0: "healthy", 1: "healthy", 2: "healthy", 3: "healthy", 4: "warning", 5: "healthy"},
 "adj": adj_healthy,
 "new_edges": [],
 "severed_edges": [],
 "proposed_edges": [],
 "show_split": False,
 "shockwave_node": None,
 "shockwave_progress": 0.0,
 "v_energy": 14.8
 },
 {
 "step": 2,
 "name": "Sensor Anomaly Spike",
 "description": "CRITICAL: Vibration Sensor enters severe oscillation. Path to Actuator Feedback shows high leakage.",
 "alarms": [4],
 "node_states": {0: "healthy", 1: "healthy", 2: "healthy", 3: "healthy", 4: "critical", 5: "warning"},
 "adj": adj_healthy,
 "new_edges": [],
 "severed_edges": [],
 "proposed_edges": [],
 "show_split": False,
 "shockwave_node": None,
 "shockwave_progress": 0.0,
 "v_energy": 28.3
 },
 {
 "step": 3,
 "name": "Orchestrator Proposes Revision",
 "description": "OCGR detects persistent structural leakage on Vibration -> Actuator. Proposing to sever link and reroute through Core Speed.",
 "alarms": [4],
 "node_states": {0: "healthy", 1: "healthy", 2: "healthy", 3: "healthy", 4: "critical", 5: "warning"},
 "adj": adj_healthy,
 "new_edges": [],
 "severed_edges": [(4, 5)],
 "proposed_edges": [(1, 5)],
 "show_split": False,
 "shockwave_node": None,
 "shockwave_progress": 0.0,
 "v_energy": 29.1
 },
 {
 "step": 4,
 "name": "Dual-World Counterfactual Split",
 "description": "Simulating two parallel worlds. World 0 (Factual, no repair) vs World 1 (Intervened, rerouted path).",
 "alarms": [4],
 "node_states": {0: "healthy", 1: "healthy", 2: "healthy", 3: "healthy", 4: "critical", 5: "counterfactual"},
 "adj": adj_healthy,
 "new_edges": [],
 "severed_edges": [(4, 5)],
 "proposed_edges": [(1, 5)],
 "show_split": True,
 "shockwave_node": 1,
 "shockwave_progress": 0.5,
 "v_energy": 29.1
 },
 {
 "step": 5,
 "name": "Lyapunov Stability Evaluation",
 "description": "Lyapunov Gate evaluates energy: V_before = 29.1, V_after = 11.2. The structural rewrite strictly lowers systemic energy.",
 "alarms": [4],
 "node_states": {0: "healthy", 1: "healthy", 2: "healthy", 3: "healthy", 4: "critical", 5: "counterfactual"},
 "adj": adj_healthy,
 "new_edges": [],
 "severed_edges": [(4, 5)],
 "proposed_edges": [(1, 5)],
 "show_split": True,
 "shockwave_node": 1,
 "shockwave_progress": 1.0,
 "v_energy": 29.1
 },
 {
 "step": 6,
 "name": "Graph Self-Repair Completed",
 "description": "ACCEPTED: Reroute path (Core Speed -> Actuator) finalized. Vibration -> Actuator permanently severed.",
 "alarms": [],
 "node_states": {i: "healthy" for i in range(N_NODES)},
 "adj": adj_repaired,
 "new_edges": [(1, 5)],
 "severed_edges": [(4, 5)],
 "proposed_edges": [],
 "show_split": False,
 "shockwave_node": None,
 "shockwave_progress": 0.0,
 "v_energy": 11.2
 },
 {
 "step": 7,
 "name": "System Re-Stabilized",
 "description": "Leakage successfully mitigated. Total energy stabilizes at 11.2. Convergence complete.",
 "alarms": [],
 "node_states": {i: "healthy" for i in range(N_NODES)},
 "adj": adj_repaired,
 "new_edges": [],
 "severed_edges": [],
 "proposed_edges": [],
 "show_split": False,
 "shockwave_node": None,
 "shockwave_progress": 0.0,
 "v_energy": 11.2
 }
]

# Generate mock counterfactual trajectories for step 4 & 5
class MockResult:
 def __init__(self):
 self.baseline_trajectory = np.random.normal(1.0, 0.1, (100, 6))
 self.baseline_trajectory[30:, 5] = np.linspace(1.0, 4.5, 70) # Cascade crash
 
 self.intervention_trajectory = np.random.normal(1.0, 0.1, (100, 6))
 # Stays stable due to repair
 self.intervention_trajectory[30:, 5] = np.random.normal(1.0, 0.05, 70)
 
 self.divergence = self.intervention_trajectory - self.baseline_trajectory
 self.affected_nodes = [5]
 self.intervention_value_score = 8.42

mock_cf_result = MockResult()

# Build Dash Application
app = Dash(__name__)
app.title = "CausalNerve Live Self-Repair Showcase"

app.layout = html.Div(style={
 'backgroundColor': THEME["bg"],
 'color': THEME["text"],
 'fontFamily': 'system-ui, sans-serif',
 'padding': '30px'
}, children=[
 html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
 html.H1("CausalNerve Cinematic Self-Repair Showcase", style={'margin': 0, 'color': '#FFFFFF'}),
 html.Div(children=[
 html.Button("⏮️ Reset", id="btn-reset", style={'padding': '8px 16px', 'marginRight': '10px', 'cursor': 'pointer', 'background': '#21262D', 'color': '#C9D1D9', 'border': '1px solid #30363D'}),
 html.Button("▶️ Next Step", id="btn-next", style={'padding': '8px 16px', 'cursor': 'pointer', 'background': '#238636', 'color': '#FFFFFF', 'border': 'none'})
 ])
 ]),
 
 html.Hr(style={'borderColor': '#30363D', 'margin': '20px 0'}),
 
 html.Div(style={'display': 'flex', 'gap': '30px'}, children=[
 # Left Panel - Status & Info
 html.Div(style={'width': '30%', 'backgroundColor': '#161B22', 'padding': '20px', 'borderRadius': '6px', 'border': '1px solid #30363D'}, children=[
 html.H3("Live Event log", style={'color': '#8B949E', 'marginTop': 0}),
 html.Div(id="step-badge", style={
 'display': 'inline-block',
 'padding': '4px 8px',
 'borderRadius': '4px',
 'fontWeight': 'bold',
 'fontSize': '12px',
 'marginBottom': '10px'
 }),
 html.H2(id="step-name", style={'margin': '0 0 15px 0', 'color': '#FFFFFF'}),
 html.P(id="step-desc", style={'fontSize': '16px', 'lineHeight': '1.5', 'color': '#C9D1D9'}),
 
 html.Hr(style={'borderColor': '#30363D', 'margin': '20px 0'}),
 
 html.H4("Lyapunov Free Energy V(G)", style={'color': '#8B949E', 'margin': '0 0 5px 0'}),
 html.Div(id="step-energy", style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#58A6FF'})
 ]),
 
 # Right Panel - Visualizations
 html.Div(style={'width': '70%', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
 html.Div(id="graph-container", children=[
 dcc.Graph(id="live-graph", config={'displayModeBar': False})
 ]),
 html.Div(id="cf-container", style={'display': 'none'}, children=[
 html.H3("Dual-World Divergence Analysis", style={'color': '#FFFFFF', 'margin': '0 0 10px 0'}),
 dcc.Graph(id="cf-split", config={'displayModeBar': False})
 ])
 ])
 ]),
 
 dcc.Store(id="current-step-store", data=0)
])

@app.callback(
 [Output("current-step-store", "data"),
 Output("step-name", "children"),
 Output("step-desc", "children"),
 Output("step-energy", "children"),
 Output("step-badge", "children"),
 Output("step-badge", "style"),
 Output("live-graph", "figure"),
 Output("cf-container", "style"),
 Output("cf-split", "figure")],
 [Input("btn-next", "n_clicks"),
 Input("btn-reset", "n_clicks")],
 [Input("current-step-store", "data")]
)
def update_step(next_clicks, reset_clicks, current_step):
 # Resolve trigger
 from dash import callback_context
 trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else None
 
 step_idx = current_step
 if "btn-reset" in str(trigger):
 step_idx = 0
 elif "btn-next" in str(trigger):
 step_idx = (step_idx + 1) % len(SCENARIO)
 
 s = SCENARIO[step_idx]
 
 # Badge colors
 badge_colors = {
 "healthy": ('#238636', '#FFFFFF'),
 "warning": ('#D29922', '#0D1117'),
 "critical": ('#F85149', '#FFFFFF'),
 "counterfactual": ('#A371F7', '#FFFFFF')
 }
 
 # Determine step status for badge
 status = "healthy"
 if s["alarms"]:
 status = "critical"
 elif "Warning" in s["name"]:
 status = "warning"
 elif s["show_split"]:
 status = "counterfactual"
 
 bg, fg = badge_colors.get(status, badge_colors["healthy"])
 badge_style = {
 'backgroundColor': bg,
 'color': fg,
 'display': 'inline-block',
 'padding': '4px 8px',
 'borderRadius': '4px',
 'fontWeight': 'bold',
 'fontSize': '12px',
 'marginBottom': '10px'
 }
 
 # Generate graph plot
 fig_graph = plot_living_graph(
 adj=s["adj"],
 node_labels=node_labels,
 node_states=s["node_states"],
 alarms=s["alarms"],
 new_edges=s["new_edges"],
 severed_edges=s["severed_edges"],
 proposed_edges=s["proposed_edges"],
 shockwave_node=s["shockwave_node"],
 shockwave_progress=s["shockwave_progress"]
 )
 
 # Handle split screen
 cf_style = {'display': 'block'} if s["show_split"] else {'display': 'none'}
 fig_cf = plot_counterfactual_split(mock_cf_result, node_labels)
 
 return (
 step_idx,
 s["name"],
 s["description"],
 f"{s['v_energy']:.1f}",
 status.upper(),
 badge_style,
 fig_graph,
 cf_style,
 fig_cf
 )

if __name__ == "__main__":
 print("Launching reproducible Live Self-Repair Showcase...")
 print("Open http://127.0.0.1:8060/ in your browser.")
 app.run(debug=True, port=8060)
