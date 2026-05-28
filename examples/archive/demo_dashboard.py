"""
demo_dashboard.py
=================
A real-time Dash application showcasing the CausalNerve viral visual engine.
"""

from dash import Dash, dcc, html, Input, Output
import numpy as np
from causalnerve.visualization_stub.graph_viz import plot_living_graph, THEME
from causalnerve.visualization_stub.counterfactual_viz import plot_counterfactual_split
from causalnerve.visualization_stub.fleet_viz import plot_fleet_recurrence

# Mock Data Generation
def generate_mock_graph():
    adj = np.zeros((10, 10))
    for i in range(9):
        adj[i, i+1] = np.random.uniform(0.3, 0.8)
    adj[2, 5] = 0.9
    adj[7, 2] = 0.5
    return adj

class MockCounterfactualResult:
    def __init__(self):
        self.affected_nodes = [2, 5, 6]
        self.unaffected_nodes = [0, 1, 3, 4, 7, 8, 9]
        self.baseline_trajectory = np.sin(np.linspace(0, 10, 100))[:, None] + np.random.normal(0, 0.1, (100, 10))
        self.intervention_trajectory = self.baseline_trajectory.copy()
        # Introduce divergence
        self.intervention_trajectory[50:, 2] += 2.0
        self.intervention_trajectory[60:, 5] += 1.5
        self.intervention_trajectory[70:, 6] += 0.8
        self.divergence = self.intervention_trajectory - self.baseline_trajectory
        self.intervention_value_score = 4.25
        self.explanation = "Severing node 2 prevents the cascade failure to node 6."

# Initialize App
app = Dash(__name__)
app.title = "CausalNerve Live Diagnostics"

node_labels = {i: f"Sensor_{i}" for i in range(10)}
adj = generate_mock_graph()
cf_res = MockCounterfactualResult()

# Precompute initial figures
fig_graph = plot_living_graph(adj, node_labels, alarms=[2, 5], new_edges=[(7, 2)], severed_edges=[(2, 5)])

cost_matrix = np.random.rand(50, 50)
path = [(i, i) for i in range(50)]
fig_fleet = plot_fleet_recurrence(cost_matrix, "Asset_A", "Historical_Crash_01", path)

fig_cf = plot_counterfactual_split(cf_res, node_labels)

# Layout
app.layout = html.Div(style={'backgroundColor': THEME["bg"], 'color': THEME["text"], 'padding': '20px', 'fontFamily': 'sans-serif'}, children=[
    html.H1("CausalNerve: Adaptive Structural Dependency Dashboard"),
    
    html.Div([
        html.Div([
            html.H3("Live Graph Orchestration"),
            dcc.Graph(id='live-graph', figure=fig_graph)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Fleet Structural Epidemiology"),
            dcc.Graph(id='fleet-map', figure=fig_fleet)
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ]),
    
    html.Div([
        html.H3("Dual-World Counterfactual Interventions (do-calculus)"),
        dcc.Graph(id='counterfactual-split', figure=fig_cf)
    ], style={'marginTop': '40px'})
])

if __name__ == '__main__':
    print("Launching CausalNerve Viral Demo Dashboard on http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)
