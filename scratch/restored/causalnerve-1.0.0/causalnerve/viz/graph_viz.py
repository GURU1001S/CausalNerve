"""
causalnerve.visualization_stub.graph_viz
=========================
High-performance visual engine for CausalNerve.
Renders living, breathing structural graphs with pulse animations, anomaly heatmaps,
and intervention shockwaves.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple

# CausalNerve Visual Language - Upgraded for Virality & Legibility
THEME = {
    "bg": "#0D1117",
    "text": "#C9D1D9",
    "grid": "#21262D",
    
    # Node Status Colors
    "healthy": "#238636",      # Green
    "warning": "#D29922",      # Amber
    "critical": "#F85149",     # Red
    "counterfactual": "#A371F7", # Purple
    
    # Edge Status Colors
    "edge_active": "#58A6FF",  # Bright Blue
    "edge_new": "#39C5BB",     # Cyan (Accepted)
    "edge_severed": "#484F58", # Gray (Rejected/Severed)
    "edge_proposed": "#FF7B72", # Light Red/Salmon (Flashing/Proposed)
}

def _get_node_positions(adj: np.ndarray, seed: int = 42):
    G = nx.DiGraph(adj)
    G.add_nodes_from(range(adj.shape[0]))
    return nx.spring_layout(G, seed=seed, k=2.0)

def plot_living_graph(adj: np.ndarray, 
                      node_labels: Dict[int, str], 
                      node_states: Dict[int, str] = None,  # dict of node_id -> 'healthy'/'warning'/'critical'/'counterfactual'
                      alarms: List[int] = None, 
                      new_edges: List[Tuple[int, int]] = None,
                      severed_edges: List[Tuple[int, int]] = None,
                      proposed_edges: List[Tuple[int, int]] = None,
                      rejected_edges: List[Tuple[int, int]] = None,
                      shockwave_node: int = None,
                      shockwave_progress: float = 0.0) -> go.Figure:
    """
    Renders the living causal/structural dependency graph with advanced visual cues
    representing self-repair, alarms, and intervention shockwaves.
    """
    node_states = node_states or {}
    alarms = alarms or []
    new_edges = new_edges or []
    severed_edges = severed_edges or []
    proposed_edges = proposed_edges or []
    rejected_edges = rejected_edges or []
    
    n_nodes = adj.shape[0]
    pos = _get_node_positions(adj)
    
    edge_traces = []
    
    # Render all edge variations
    for u in range(n_nodes):
        for v in range(n_nodes):
            w = adj[u, v]
            edge = (u, v)
            
            is_severed = edge in severed_edges
            is_new = edge in new_edges
            is_proposed = edge in proposed_edges
            is_rejected = edge in rejected_edges
            
            if w > 0 or is_severed or is_proposed or is_rejected:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Determine colors and styles based on state
                if is_severed or is_rejected:
                    color = THEME["edge_severed"]
                    width = 1.5
                    dash = "dash"
                elif is_proposed:
                    color = THEME["edge_proposed"]
                    width = 3.5
                    dash = "dot"
                elif is_new:
                    color = THEME["edge_new"]
                    width = 4.0
                    dash = "solid"
                else:
                    color = THEME["edge_active"]
                    width = max(1.5, w * 5.0)
                    dash = "solid"
                
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=width, color=color, dash=dash),
                        mode='lines',
                        hoverinfo='none',
                        showlegend=False
                    )
                )
                
    # Node coordinates and styles
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_borders = []
    node_border_colors = []
    
    for i in range(n_nodes):
        x, y = pos[i]
        node_x.append(x)
        node_y.append(y)
        
        # Resolve node health/status color
        state = node_states.get(i, "healthy")
        if i in alarms:
            state = "critical"
            
        color = THEME.get(state, THEME["healthy"])
        node_colors.append(color)
        
        # Active warning/critical nodes pulse larger
        if state == "critical":
            size = 32
            border_w = 3
            border_c = "#FFFFFF"
        elif state == "warning":
            size = 26
            border_w = 2
            border_c = THEME["warning"]
        elif state == "counterfactual":
            size = 28
            border_w = 2.5
            border_c = "#FFFFFF"
        else:
            size = 20
            border_w = 1.5
            border_c = THEME["bg"]
            
        node_sizes.append(size)
        node_borders.append(border_w)
        node_border_colors.append(border_c)
        node_text.append(node_labels.get(i, f"Node {i}"))
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(color=THEME["text"], size=12, family="sans-serif"),
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=node_borders, color=node_border_colors)
        )
    )
    
    # Optional: Render Intervention Shockwave Effect
    shapes = []
    if shockwave_node is not None and shockwave_progress > 0:
        sx, sy = pos[shockwave_node]
        radius = shockwave_progress * 1.5  # Expands over time
        shapes.append(dict(
            type="circle",
            xref="x", yref="y",
            x0=sx - radius, y0=sy - radius,
            x1=sx + radius, y1=sy + radius,
            line_color=THEME["counterfactual"],
            line_width=3,
            fillcolor="rgba(163, 113, 247, 0.1)"
        ))

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=10, r=10, t=40),
            plot_bgcolor=THEME["bg"],
            paper_bgcolor=THEME["bg"],
            font=dict(color=THEME["text"]),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            shapes=shapes
        )
    )
    return fig

def create_timeline_replay(history: List[Dict]) -> go.Figure:
    """
    Placeholder for Plotly frames timeline replay.
    """
    pass
