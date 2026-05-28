"""
causalnerve.visualization_stub.counterfactual_viz
==================================
Renders the dual-world split screen: Factual vs. Intervened future.
Shows divergence dynamically.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any

# Reusing the theme
THEME = {
    "bg": "#0D1117",
    "factual": "#8B949E", # Muted gray
    "intervened": "#58A6FF", # Bright blue
    "divergence": "#A371F7", # Purple
    "text": "#C9D1D9",
    "grid": "#30363D"
}

def plot_counterfactual_split(result: Any, node_labels: Dict[int, str]) -> go.Figure:
    """
    Renders the split-screen view.
    Left: Factual (World 0)
    Right: Intervened (World 1)
    """
    fig = make_subplots(rows=2, cols=2, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        horizontal_spacing=0.05,
                        subplot_titles=("Factual Future (World 0)", "Intervened Future (World 1)", "Divergence (W1 - W0)"),
                        specs=[[{"type": "xy"}, {"type": "xy"}],
                               [{"type": "xy", "colspan": 2}, None]])

    T = result.baseline_trajectory.shape[0]
    time_axis = np.arange(T)
    
    # We plot the top affected nodes
    affected = result.affected_nodes[:3] # Top 3 for clarity
    
    # 1. Factual Trajectories (Top Left)
    for node in affected:
        name = node_labels.get(node, f"N{node}")
        fig.add_trace(go.Scatter(x=time_axis, y=result.baseline_trajectory[:, node], 
                                 mode='lines', name=f"{name} (Factual)",
                                 line=dict(color=THEME["factual"], dash="dash")), row=1, col=1)

    # 2. Intervened Trajectories (Top Right)
    for node in affected:
        name = node_labels.get(node, f"N{node}")
        fig.add_trace(go.Scatter(x=time_axis, y=result.intervention_trajectory[:, node], 
                                 mode='lines', name=f"{name} (Intervened)",
                                 line=dict(color=THEME["intervened"])), row=1, col=2)
                                 
    # 3. Divergence Curve (Bottom)
    for node in affected:
        name = node_labels.get(node, f"N{node}")
        div = result.divergence[:, node]
        fig.add_trace(go.Scatter(x=time_axis, y=div, 
                                 mode='lines', name=f"{name} (Divergence)",
                                 line=dict(color=THEME["divergence"])), row=2, col=1)
                                 
    # Fill between for total divergence volume
    total_div = np.sum(np.abs(result.divergence[:, affected]), axis=1)
    fig.add_trace(go.Scatter(x=time_axis, y=total_div, 
                             mode='lines', fill='tozeroy', name="Total Divergence Vol",
                             line=dict(color="rgba(163, 113, 247, 0.2)")), row=2, col=1)

    # Styling
    fig.update_layout(
        title=f"Counterfactual Intervention Analysis<br><sup>Intervention Value Score: {result.intervention_value_score:.2f}</sup>",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=THEME["grid"])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=THEME["grid"])

    return fig
