import json
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

@dataclass
class GraphDiff:
    edges_added: List[Tuple[int, int]] = field(default_factory=list)
    edges_removed: List[Tuple[int, int]] = field(default_factory=list)
    edges_stable: List[Tuple[int, int]] = field(default_factory=list)
    confidence_changes: Dict[Tuple[int, int], float] = field(default_factory=dict)

@dataclass
class GraphSnapshot:
    cycle: int
    adjacency: List[List[Any]] # e.g. [[u, v, weight], ...]
    leakage: float
    v_energy: float
    active_alarms: List[int]
    plotly_json: str # pre-rendered graph figure

@dataclass
class RevisionEvent:
    cycle: int
    edit_type: str
    edge: Tuple[int, int]
    confidence: float
    rationale: str

class StructuralReplayEngine:
    """
    Records graph topology snapshots over time and enables
    scrubbing backward to inspect the graph at any past cycle.
    
    This transforms the dashboard from a live monitor into
    a time-travel causal debugger.
    """
    
    def __init__(self,
                  snapshot_interval: int = 10,
                  max_snapshots: int = 1000):
        """
        snapshot_interval: record one snapshot every N cycles
        max_snapshots: maximum snapshots to keep in memory
                       (older snapshots are evicted FIFO)
        """
        self.snapshots: List[GraphSnapshot] = []
        self.revision_events: List[RevisionEvent] = []
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
    
    def record_snapshot(self,
                          cycle: int,
                          adjacency_matrix: List[List[Any]],
                          leakage: float,
                          v_energy: float,
                          active_alarms: List[int],
                          node_labels: Dict[int, str]
                          ) -> None:
        """
        Called by OCGR every snapshot_interval cycles.
        
        Stores:
            cycle, adjacency_matrix (sparse),
            leakage, v_energy, active_alarms
        """
        # Pre-render graph to meet the <500ms constraint
        fig = self._render_base_graph(adjacency_matrix, active_alarms, node_labels)
        plotly_json = fig.to_json()
        
        snapshot = GraphSnapshot(
            cycle=cycle,
            adjacency=adjacency_matrix,
            leakage=leakage,
            v_energy=v_energy,
            active_alarms=active_alarms,
            plotly_json=plotly_json
        )
        
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
    
    def record_revision(self, event: RevisionEvent) -> None:
        """Record every revision event with its cycle."""
        self.revision_events.append(event)
    
    def get_snapshot(self, cycle: int) -> Optional[GraphSnapshot]:
        """
        Return the snapshot closest to `cycle` (specifically <= cycle).
        If cycle is between two snapshots: return the earlier one.
        """
        if not self.snapshots:
            return None
            
        best = self.snapshots[0]
        for s in self.snapshots:
            if s.cycle <= cycle:
                best = s
            else:
                break
        return best
    
    def get_graph_diff(self,
                        cycle_a: int,
                        cycle_b: int
                        ) -> GraphDiff:
        """
        Returns the structural diff between two cycles:
            edges_added: edges in cycle_b but not cycle_a
            edges_removed: edges in cycle_a but not cycle_b
            edges_stable: edges in both
            confidence_changes: edges whose confidence changed
        
        Used by the timeline scrubber to highlight changes.
        """
        snap_a = self.get_snapshot(cycle_a)
        snap_b = self.get_snapshot(cycle_b)
        
        diff = GraphDiff()
        if not snap_a or not snap_b:
            return diff
            
        edges_a = {(e[0], e[1]): e[2] for e in snap_a.adjacency}
        edges_b = {(e[0], e[1]): e[2] for e in snap_b.adjacency}
        
        for e in edges_b:
            if e not in edges_a:
                diff.edges_added.append(e)
            else:
                diff.edges_stable.append(e)
                if abs(edges_b[e] - edges_a[e]) > 1e-4:
                    diff.confidence_changes[e] = edges_b[e] - edges_a[e]
                    
        for e in edges_a:
            if e not in edges_b:
                diff.edges_removed.append(e)
                
        return diff
    
    def get_revision_events_in_range(self,
                                      start_cycle: int,
                                      end_cycle: int
                                      ) -> List[RevisionEvent]:
        """Returns all revision events between start and end cycles."""
        return [r for r in self.revision_events if start_cycle <= r.cycle <= end_cycle]
    
    def export_replay(self, path: str, preset: str, n_nodes: int, node_labels: Dict[int, str]) -> None:
        """
        Export full replay data to a .causalnerve-replay file.
        Format: JSON with all snapshots and revision events.
        """
        data = {
            "version": "0.1",
            "preset": preset,
            "n_nodes": n_nodes,
            "node_labels": node_labels,
            "total_cycles": self.snapshots[-1].cycle if self.snapshots else 0,
            "snapshots": [
                {
                    "cycle": s.cycle,
                    "adjacency": s.adjacency,
                    "leakage": s.leakage,
                    "v_energy": s.v_energy,
                    "active_alarms": s.active_alarms
                } for s in self.snapshots
            ],
            "revision_events": [
                {
                    "cycle": r.cycle,
                    "edit_type": r.edit_type,
                    "edge": list(r.edge),
                    "confidence": r.confidence,
                    "rationale": r.rationale
                } for r in self.revision_events
            ]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_replay(cls, path: str) -> 'StructuralReplayEngine':
        """Load a previously exported replay for analysis."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        engine = cls()
        node_labels = {int(k): v for k, v in data.get("node_labels", {}).items()}
        
        for s_data in data.get("snapshots", []):
            engine.record_snapshot(
                cycle=s_data["cycle"],
                adjacency_matrix=s_data["adjacency"],
                leakage=s_data["leakage"],
                v_energy=s_data["v_energy"],
                active_alarms=s_data.get("active_alarms", []),
                node_labels=node_labels
            )
            
        for r_data in data.get("revision_events", []):
            engine.record_revision(RevisionEvent(
                cycle=r_data["cycle"],
                edit_type=r_data["edit_type"],
                edge=tuple(r_data["edge"]),
                confidence=r_data.get("confidence", 0.0),
                rationale=r_data.get("rationale", "")
            ))
            
        return engine

    def _render_base_graph(self, adjacency, active_alarms, node_labels) -> go.Figure:
        """
        Pre-renders the base Plotly figure for a snapshot.
        """
        fig = go.Figure()
        
        # Determine nodes from adjacency or node_labels
        nodes = set(node_labels.keys())
        for u, v, w in adjacency:
            nodes.add(u)
            nodes.add(v)
            
        # Basic circular layout
        n_nodes = len(nodes)
        if n_nodes == 0:
            return fig
            
        theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        pos = {n: (np.cos(th), np.sin(th)) for n, th in zip(sorted(list(nodes)), theta)}
        
        # Add edges
        for u, v, w in adjacency:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=max(1, w*5), color='blue'),
                mode='lines',
                hoverinfo='none',
                showlegend=False
            ))
            
        # Add nodes
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        node_text = [node_labels.get(n, f"Node {n}") for n in nodes]
        node_color = ['red' if n in active_alarms else 'green' for n in nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig

    def render_diff_graph(self, snapshot: GraphSnapshot, diff: GraphDiff, current_alarms: List[int], node_labels: Dict[int, str]) -> go.Figure:
        """
        Renders graph with historical diff highlighting (used by scrubber).
        Edges stable: blue
        Edges removed: amber dashed
        Edges added: teal dashed
        """
        # Start from base Plotly object in snapshot
        fig = go.Figure(json.loads(snapshot.plotly_json))
        
        # The base graph only has edges that existed at that cycle (which includes stable and removed).
        # We need to recolor removed edges to amber dashed.
        # We need to add 'edges added' as teal dashed.
        
        # To do this cleanly, it's often easier to just redraw edges
        fig.data = [] # Clear traces
        
        nodes = set(node_labels.keys())
        for e in diff.edges_stable + diff.edges_removed + diff.edges_added:
            nodes.add(e[0])
            nodes.add(e[1])
            
        n_nodes = len(nodes)
        if n_nodes == 0:
            return fig
            
        theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        pos = {n: (np.cos(th), np.sin(th)) for n, th in zip(sorted(list(nodes)), theta)}
        
        # 1. Stable edges
        for u, v in diff.edges_stable:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=2, color='blue'),
                mode='lines', showlegend=False
            ))
            
        # 2. Removed edges (existed then, removed since)
        for u, v in diff.edges_removed:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=2, color='orange', dash='dash'),
                mode='lines', showlegend=False
            ))
            
        # 3. Added edges (exist now, did not exist then)
        for u, v in diff.edges_added:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=2, color='teal', dash='dash'),
                mode='lines', showlegend=False
            ))
            
        # 4. Nodes
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        node_text = [node_labels.get(n, f"Node {n}") for n in nodes]
        
        # Split color: half then, half now. (Simulated by using a marker line color for 'then' and fill for 'now')
        # Fill: now
        # Line: then
        node_color = ['red' if n in current_alarms else 'green' for n in nodes]
        node_line = ['red' if n in snapshot.active_alarms else 'green' for n in nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color=node_color,
                line=dict(width=4, color=node_line)
            ),
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
