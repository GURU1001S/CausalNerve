import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FleetEpidemiologyVisualizer:
    def __init__(self, num_engines: int = 20):
        self.num_engines = num_engines
        self.bg_color = "#0d1117"
        
        self.fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]],
            subplot_titles=("Live Fleet Degradation Heatmap (Engines vs Cycles)", 
                            "Cross-Fleet Root Cause Mining",
                            "Cumulative Fleet Risk (Engines > Threshold)", 
                            "Engines at Critical Risk (<50 Cycles to Failure)")
        )
        
        # Panel A: Fleet Heatmap
        # X: Cycle, Y: Engine ID, Z: Degradation state
        self.fig.add_trace(go.Heatmap(z=[], x=[], y=[], colorscale="Turbo", showscale=True), row=1, col=1)
        
        # Panel B: Motif Mining Bar Chart
        self.fig.add_trace(go.Bar(x=[], y=[], marker_color="#a371f7", orientation='h'), row=1, col=2)
        
        # Panel C: Fleet Risk Scatter (Number of engines above critical threshold over time)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color="#f85149", width=3), fill='tozeroy'), row=2, col=1)
        
        # Panel D: Critical Risk Table
        self.fig.add_trace(go.Table(
            header=dict(values=["Engine ID", "Est. Cycles to Failure", "Primary Motif"],
                        fill_color="#21262d", font=dict(color="white", size=12)),
            cells=dict(values=[[], [], []], fill_color="#0d1117", font=dict(color="white", size=11))
        ), row=2, col=2)
        
        self.fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            showlegend=False,
            height=800,
            width=1400,
            margin=dict(l=40, r=40, t=60, b=40),
            title="NASA C-MAPSS: Fleet Epidemiology & Predictive Intelligence Dashboard"
        )
        
        self._html_frames = []
        self.heatmap_z = [[] for _ in range(self.num_engines)]
        self.cycles = []
        self.risk_count = []
        
    def update(self, cycle: int, engine_healths: list, motifs: dict, risk_table: list):
        self.cycles.append(cycle)
        
        # Heatmap update
        critical_count = 0
        for i in range(self.num_engines):
            h = engine_healths[i]
            self.heatmap_z[i].append(h)
            if h > 0.7: critical_count += 1
            
        self.risk_count.append(critical_count)
        
        # Motifs update
        m_labels = list(motifs.keys())
        m_vals = list(motifs.values())
        
        # Table update
        t_ids = [r[0] for r in risk_table]
        t_rul = [r[1] for r in risk_table]
        t_cause = [r[2] for r in risk_table]
        
        frame_data = [
            go.Heatmap(z=self.heatmap_z, x=self.cycles, y=[f"E-{i+1}" for i in range(self.num_engines)]),
            go.Bar(x=m_vals, y=m_labels, marker_color="#a371f7", orientation='h'),
            go.Scatter(x=self.cycles, y=self.risk_count, mode='lines', line=dict(color="#f85149", width=3), fill='tozeroy'),
            go.Table(
                header=dict(values=["Engine ID", "Est. Cycles to Failure", "Primary Motif"],
                            fill_color="#21262d", font=dict(color="white", size=12)),
                cells=dict(values=[t_ids, t_rul, t_cause], fill_color="#0d1117", font=dict(color="white", size=11))
            )
        ]
        
        self._html_frames.append(go.Frame(data=frame_data, name=f"Cycle {cycle}"))

    def render_animation_html(self, path: str = "fleet_epidemiology.html"):
        if not self._html_frames: return
        self.fig.frames = self._html_frames
        
        self.fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=-0.1,
                x=0.0,
                xanchor="left",
                yanchor="bottom",
                buttons=[dict(
                    label="Play Fleet Stream",
                    method="animate",
                    args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, transition=dict(duration=0))]
                )]
            )]
        )
        self.fig.write_html(path, auto_play=False)
        print(f"[Visualizer] Fleet Epidemiology dashboard compiled to: {path}")
