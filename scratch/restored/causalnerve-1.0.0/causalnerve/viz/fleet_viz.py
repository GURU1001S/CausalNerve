"""
causalnerve.visualization_stub.fleet_viz
=========================
Visualizes Fleet Structural Epidemiology.
Plots trajectory alignment (DTW) and outbreak recurrence maps across multiple assets.
"""

import plotly.graph_objects as go
import numpy as np

THEME = {
    "bg": "#0D1117",
    "text": "#C9D1D9",
    "heatmap_colorscale": "Inferno"
}

def plot_fleet_recurrence(cost_matrix: np.ndarray, asset_id: str, match_id: str, path: list = None) -> go.Figure:
    """
    Renders the Dynamic Time Warping (DTW) structural recurrence matrix.
    Shows how the current asset's structural degradation matches a historical outbreak.
    """
    fig = go.Figure(data=go.Heatmap(
                   z=cost_matrix,
                   colorscale=THEME["heatmap_colorscale"],
                   hoverongaps=False))
                   
    # Plot the optimal warp path if provided
    if path:
        x_path = [p[1] for p in path]
        y_path = [p[0] for p in path]
        
        fig.add_trace(go.Scatter(
            x=x_path, y=y_path,
            mode='lines',
            line=dict(color="cyan", width=3),
            name="Optimal Alignment Path"
        ))

    fig.update_layout(
        title=f"Fleet Structural Epidemiology<br><sup>Alignment: {asset_id} (Current) vs {match_id} (Historical Outbreak)</sup>",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        xaxis_title=f"{match_id} Structural History",
        yaxis_title=f"{asset_id} Structural History",
        height=600,
        width=800
    )
    return fig
