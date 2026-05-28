"""
export_demo_gifs.py
===================
Automatic viral-content export engine.
Generates self_repair.gif, counterfactual_split.gif, and fleet_prediction.gif.
Requires: pip install kaleido imageio
"""

import os
import time
import numpy as np
import imageio.v2 as imageio
from typing import Dict, List, Tuple

from causalnerve.visualization_stub.graph_viz import plot_living_graph, THEME
from causalnerve.visualization_stub.counterfactual_viz import plot_counterfactual_split
from causalnerve.visualization_stub.fleet_viz import plot_fleet_recurrence

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# Shared Mock metadata
node_labels = {
    0: "Fuel Flow",
    1: "Core Speed",
    2: "Exhaust Temp",
    3: "Oil Pressure",
    4: "Vibration Sensor",
    5: "Actuator Feedback"
}

def export_self_repair_gif():
    print("Generating self_repair.gif...")
    frames = []
    
    # Define a 10-frame animation cycle
    n_frames = 15
    adj_base = np.zeros((6, 6))
    adj_base[0, 1] = 0.8
    adj_base[1, 2] = 0.7
    adj_base[1, 3] = 0.6
    adj_base[0, 4] = 0.5
    adj_base[4, 5] = 0.7
    
    for f in range(n_frames):
        # Progress story
        alarms = []
        node_states = {i: "healthy" for i in range(6)}
        adj = adj_base.copy()
        new_edges = []
        severed_edges = []
        proposed_edges = []
        shockwave_node = None
        shockwave_progress = 0.0
        
        if 2 <= f < 5:
            # Alarm
            alarms = [4]
            node_states[4] = "critical"
            node_states[5] = "warning"
        elif 5 <= f < 8:
            # Revision proposed
            alarms = [4]
            node_states[4] = "critical"
            node_states[5] = "warning"
            severed_edges = [(4, 5)]
            proposed_edges = [(1, 5)]
        elif 8 <= f < 11:
            # Shockwave / Counterfactual split active
            alarms = [4]
            node_states[4] = "critical"
            node_states[5] = "counterfactual"
            severed_edges = [(4, 5)]
            proposed_edges = [(1, 5)]
            shockwave_node = 1
            shockwave_progress = (f - 8) / 3.0
        elif f >= 11:
            # Repaired
            adj[4, 5] = 0.0
            adj[1, 5] = 0.75
            new_edges = [(1, 5)]
            severed_edges = [(4, 5)]
            
        fig = plot_living_graph(
            adj=adj,
            node_labels=node_labels,
            node_states=node_states,
            alarms=alarms,
            new_edges=new_edges,
            severed_edges=severed_edges,
            proposed_edges=proposed_edges,
            shockwave_node=shockwave_node,
            shockwave_progress=shockwave_progress
        )
        
        # Save temporary frame
        temp_filename = f"temp_self_repair_{f}.png"
        fig.write_image(temp_filename, scale=1.5, width=600, height=450)
        frames.append(imageio.imread(temp_filename))
        os.remove(temp_filename)
        
    imageio.mimsave("self_repair.gif", frames, fps=2)
    # Save a copy in assets
    imageio.mimsave("assets/self_repair.gif", frames, fps=2)
    print("Exported self_repair.gif successfully.")

def export_counterfactual_gif():
    print("Generating counterfactual_split.gif...")
    frames = []
    n_frames = 12
    
    class MockResult:
        def __init__(self, t_limit):
            self.baseline_trajectory = np.random.normal(1.0, 0.1, (t_limit, 6))
            self.baseline_trajectory[30:, 5] = np.linspace(1.0, 4.5, t_limit - 30)
            
            self.intervention_trajectory = np.random.normal(1.0, 0.1, (t_limit, 6))
            self.intervention_trajectory[30:, 5] = np.random.normal(1.0, 0.05, t_limit - 30)
            
            self.divergence = self.intervention_trajectory - self.baseline_trajectory
            self.affected_nodes = [5]
            self.intervention_value_score = 8.42

    for f in range(n_frames):
        t_limit = int(40 + (f / float(n_frames)) * 60)
        res = MockResult(t_limit)
        fig = plot_counterfactual_split(res, node_labels)
        
        temp_filename = f"temp_cf_{f}.png"
        fig.write_image(temp_filename, scale=1.5, width=800, height=500)
        frames.append(imageio.imread(temp_filename))
        os.remove(temp_filename)
        
    imageio.mimsave("counterfactual_split.gif", frames, fps=3)
    imageio.mimsave("assets/counterfactual_split.gif", frames, fps=3)
    print("Exported counterfactual_split.gif successfully.")

def export_fleet_gif():
    print("Generating fleet_prediction.gif...")
    frames = []
    n_frames = 10
    
    for f in range(n_frames):
        # Shift cost matrix to simulate scanning
        cost_matrix = np.roll(np.random.rand(40, 40), shift=f, axis=0)
        path = [(i, (i + f) % 40) for i in range(40)]
        fig = plot_fleet_recurrence(cost_matrix, "Asset_42", "Historical_Crash_18", path)
        
        # Add prediction label overlay
        fig.update_layout(
            title=f"Fleet Structural Epidemiology<br><sup>Predicted revision in ~{max(1, 10 - f)} cycles</sup>"
        )
        
        temp_filename = f"temp_fleet_{f}.png"
        fig.write_image(temp_filename, scale=1.5, width=700, height=450)
        frames.append(imageio.imread(temp_filename))
        os.remove(temp_filename)
        
    imageio.mimsave("fleet_prediction.gif", frames, fps=2)
    imageio.mimsave("assets/fleet_prediction.gif", frames, fps=2)
    print("Exported fleet_prediction.gif successfully.")

if __name__ == "__main__":
    export_self_repair_gif()
    export_counterfactual_gif()
    export_fleet_gif()
    print("ALL VIRAL GIFS EXPORTED SUCCESSFULLY.")
