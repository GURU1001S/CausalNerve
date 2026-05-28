import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LiveGraphVisualizer:
    def __init__(self, num_nodes: int = 24, node_labels: dict = None):
        self.num_nodes = num_nodes
        self.node_labels = node_labels or {i: f"N{i}" for i in range(num_nodes)}
        
        # Theme colors
        self.bg_color = "#0d1117"
        self.color_healthy = "#2ea043" # Green
        self.color_warning = "#d29922" # Amber
        self.color_alarm = "#f85149"   # Red
        self.color_repaired = "#3fb950"
        self.color_intervention = "#a371f7"
        self.color_cyan = "#2f81f7"    # Cyan for structural drift
        
        self.fig = make_subplots(
            rows=4, cols=3,
            column_widths=[0.35, 0.35, 0.3],
            specs=[[{"type": "scatter", "rowspan": 4}, {"type": "scatter", "rowspan": 4}, {"type": "scatter"}],
                   [None, None, {"type": "scatter"}],
                   [None, None, {"type": "scatter"}],
                   [None, None, {"type": "scatter"}]],
            subplot_titles=("WORLD 0: Factual Engine", 
                            "WORLD 1: Intervened/Repaired Engine",
                            "System Leakage L(G)", 
                            "Intervention Utility U", 
                            "Divergence Dynamics D(t)", 
                            "Intelligence Audit: V(G)")
        )
        
        # Pre-calculate node positions (circular layout)
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        self.node_x = np.cos(angles)
        self.node_y = np.sin(angles)
        
        # WORLD 0 TRACES
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='rgba(255,255,255,0.15)'), hoverinfo='none'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, color='rgba(255,255,255,0.7)'), hoverinfo='none'), row=1, col=1)
        self.fig.add_trace(go.Scatter(
            x=self.node_x, y=self.node_y, mode='markers+text',
            text=[self.node_labels.get(i, f"N{i}") for i in range(num_nodes)],
            textposition="bottom center",
            marker=dict(size=20, color=self.color_healthy, line=dict(width=2, color='white')),
            hoverinfo='text'
        ), row=1, col=1)

        # WORLD 1 TRACES
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='rgba(47,129,247,0.15)'), hoverinfo='none'), row=1, col=2)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, color='rgba(47,129,247,0.7)'), hoverinfo='none'), row=1, col=2)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, color='rgba(163,113,247,0.9)', dash='dash'), hoverinfo='none'), row=1, col=2)
        self.fig.add_trace(go.Scatter(
            x=self.node_x, y=self.node_y, mode='markers+text',
            text=[self.node_labels.get(i, f"N{i}") for i in range(num_nodes)],
            textposition="bottom center",
            marker=dict(size=20, color=self.color_repaired, line=dict(width=2, color='white')),
            hoverinfo='text'
        ), row=1, col=2)
        
        # METRICS TRACES
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=self.color_warning), name="Leakage L(G)"), row=1, col=3)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=self.color_healthy), name="Utility U"), row=2, col=3)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=self.color_intervention), name="Cumulative D(t)"), row=3, col=3)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#ff7b72', dash='dash'), name="Inst. D(t)"), row=3, col=3)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#d2a8ff', dash='dot'), name="Accel D''(t)"), row=3, col=3)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#58a6ff'), name="Lyapunov V(G)"), row=4, col=3)
        
        self.fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            showlegend=False,
            height=900,
            width=1600,
            margin=dict(l=20, r=20, t=60, b=20),
            title="NASA C-MAPSS: True Intelligence Audit & Dual-World Simulator"
        )
        
        # Hide axes for graphs
        self.fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=1, col=1)
        self.fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=1, col=1)
        self.fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=1, col=2)
        self.fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=1, col=2)
        
        self.cycles = []
        self.leakage = []
        self.gain = []
        self.cf_divergence = []
        self.cf_div_inst = []
        self.cf_div_accel = []
        self.energy = []
        self._html_frames = []

    def update(self, cycle: int, 
               adj_w0: np.ndarray, adj_w1: np.ndarray, 
               leakage: float, energy: float, gain: float, cf_div: float, 
               alarms: list, accepted_surgeries: list, 
               node_healths_w0: dict = None, node_healths_w1: dict = None, 
               root_cause_text: str = "", audit_text: str = "",
               repaired_edges_w1: list = None,
               cf_div_inst: float = 0.0, cf_div_accel: float = 0.0):
        
        self.cycles.append(cycle)
        self.leakage.append(leakage)
        self.gain.append(gain)
        self.cf_divergence.append(cf_div)
        self.cf_div_inst.append(cf_div_inst)
        self.cf_div_accel.append(cf_div_accel)
        self.energy.append(energy)
        
        # WORLD 0 Node Colors
        w0_colors = [self.color_healthy] * self.num_nodes
        w0_sizes = [20] * self.num_nodes
        if node_healths_w0:
            for idx, h in node_healths_w0.items():
                if idx < self.num_nodes:
                    if h == 'cyan':
                        w0_colors[idx] = self.color_cyan
                        w0_sizes[idx] = 22
                    elif h == 'amber':
                        w0_colors[idx] = self.color_warning
                        w0_sizes[idx] = 25
                    elif h == 'red':
                        w0_colors[idx] = self.color_alarm
                        w0_sizes[idx] = 30
                    elif h == 'repaired':
                        w0_colors[idx] = self.color_repaired
                        w0_sizes[idx] = 30
        
        # WORLD 1 Node Colors
        w1_colors = [self.color_healthy] * self.num_nodes
        w1_sizes = [20] * self.num_nodes
        if node_healths_w1:
            for idx, h in node_healths_w1.items():
                if idx < self.num_nodes:
                    if h == 'cyan':
                        w1_colors[idx] = self.color_cyan
                        w1_sizes[idx] = 25
                    elif h == 'repaired':
                        w1_colors[idx] = self.color_repaired
                        w1_sizes[idx] = 25
                    else:
                        w1_colors[idx] = self.color_healthy
        
        # Edges W0
        w0_weak_x, w0_weak_y, w0_strong_x, w0_strong_y = [], [], [], []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                w = adj_w0[i, j]
                if w > 0.1: # Adaptive sparsification
                    if w > 0.4:
                        w0_strong_x.extend([self.node_x[i], self.node_x[j], None])
                        w0_strong_y.extend([self.node_y[i], self.node_y[j], None])
                    else:
                        w0_weak_x.extend([self.node_x[i], self.node_x[j], None])
                        w0_weak_y.extend([self.node_y[i], self.node_y[j], None])
                        
        # Edges W1
        w1_weak_x, w1_weak_y, w1_strong_x, w1_strong_y, w1_rep_x, w1_rep_y = [], [], [], [], [], []
        repaired_edges_w1 = repaired_edges_w1 or []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                w = adj_w1[i, j]
                if (i, j) in repaired_edges_w1:
                    w1_rep_x.extend([self.node_x[i], self.node_x[j], None])
                    w1_rep_y.extend([self.node_y[i], self.node_y[j], None])
                elif w > 0.1:
                    if w > 0.4:
                        w1_strong_x.extend([self.node_x[i], self.node_x[j], None])
                        w1_strong_y.extend([self.node_y[i], self.node_y[j], None])
                    else:
                        w1_weak_x.extend([self.node_x[i], self.node_x[j], None])
                        w1_weak_y.extend([self.node_y[i], self.node_y[j], None])
                        
        # Update annotations for root cause & audit
        annotations = []
        if root_cause_text:
            annotations.append(
                dict(x=0.02, y=0.98, xref="paper", yref="paper", text=root_cause_text, 
                     showarrow=False, font=dict(color="white", size=14), bgcolor="rgba(0,0,0,0.6)")
            )
        if audit_text:
            annotations.append(
                dict(x=0.35, y=0.98, xref="paper", yref="paper", text=audit_text, 
                     showarrow=False, font=dict(color="white", size=12), bgcolor="rgba(47,129,247,0.4)")
            )
            
        layout_update = dict(annotations=annotations)

        frame_data = [
            go.Scatter(x=w0_weak_x, y=w0_weak_y, mode='lines', line=dict(width=1, color='rgba(255,255,255,0.15)')),
            go.Scatter(x=w0_strong_x, y=w0_strong_y, mode='lines', line=dict(width=3, color='rgba(255,255,255,0.7)')),
            go.Scatter(x=self.node_x, y=self.node_y, mode='markers+text', marker=dict(size=w0_sizes, color=w0_colors)),
            
            go.Scatter(x=w1_weak_x, y=w1_weak_y, mode='lines', line=dict(width=1, color='rgba(47,129,247,0.15)')),
            go.Scatter(x=w1_strong_x, y=w1_strong_y, mode='lines', line=dict(width=3, color='rgba(47,129,247,0.7)')),
            go.Scatter(x=w1_rep_x, y=w1_rep_y, mode='lines', line=dict(width=3, color='rgba(163,113,247,0.9)', dash='dash')),
            go.Scatter(x=self.node_x, y=self.node_y, mode='markers+text', marker=dict(size=w1_sizes, color=w1_colors)),
            
            go.Scatter(x=list(self.cycles), y=list(self.leakage), mode='lines'),
            go.Scatter(x=list(self.cycles), y=list(self.gain), mode='lines'),
            go.Scatter(x=list(self.cycles), y=list(self.cf_divergence), mode='lines'),
            go.Scatter(x=list(self.cycles), y=list(self.cf_div_inst), mode='lines'),
            go.Scatter(x=list(self.cycles), y=list(self.cf_div_accel), mode='lines'),
            go.Scatter(x=list(self.cycles), y=list(self.energy), mode='lines')
        ]
        
        self._html_frames.append(go.Frame(data=frame_data, name=f"Cycle {cycle}", layout=layout_update))

    def render_animation_html(self, path: str = "live_graph_evolution.html"):
        if not self._html_frames: return
        self.fig.frames = self._html_frames
        
        self.fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=-0.15,
                x=0.0,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Play Stream",
                        method="animate",
                        args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                    )
                ]
            )]
        )
        self.fig.write_html(path, auto_play=False)
        print(f"[Visualizer] Continuous rendering compiled to: {path}")
