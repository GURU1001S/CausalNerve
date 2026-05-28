"""
causalnerve.api
===============
Main SDK entry point. Every API call executes real graph mathematics.

Wiring:
    CausalNerve.why()                -> CausalTracer (weighted DFS)
    CausalNerve.what_if()            -> CounterfactualEngine (dual-world sim)
    CausalNerve.do()                 -> InterventionEngine (Pearl do-calculus)
    CausalNerve.watch()              -> Streaming alarm + graph revision loop
    CausalNerve.predict_next_change() -> Fleet recurrence memory
"""

from typing import Optional, Dict, List, Union, Any, Callable
import numpy as np
import time
import warnings
import json
from collections import deque
from dataclasses import dataclass, field, asdict

from causalnerve.plugins.registry import PluginRegistry
from causalnerve.plugins.interfaces import DomainPlugin
from causalnerve.interventions.intervention import CausalGraph, InterventionEngine
from causalnerve.interventions.counterfactual import CounterfactualEngine
from causalnerve.interventions.trace import CausalTracer
from causalnerve.reasoning.explanation import CausalNarrator
from causalnerve.fleet.epidemiology import FleetEpidemiologyGraph, EpidemiologyEngine, FleetRiskForecaster, TransferLearningLayer


# ─────────────────────────────────────────────────────
#  Type validation helpers
# ─────────────────────────────────────────────────────

class _Validate:
    """Strict input validation — no silent failures."""

    @staticmethod
    def node(node: Any, name_map: Dict[str, int], n_nodes: int) -> int:
        if isinstance(node, int):
            if node < 0 or node >= n_nodes:
                raise IndexError(f"Node index {node} out of range [0, {n_nodes})")
            return node
        if isinstance(node, str):
            if node in name_map:
                return name_map[node]
            raise ValueError(f"Unknown node '{node}'. Available: {sorted(name_map.keys())}")
        raise TypeError(f"Node must be int or str, got {type(node).__name__}")

    @staticmethod
    def value(value: Any, name: str = "value") -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
        return v

    @staticmethod
    def intervention_dict(d: Any, name_map: Dict[str, int], n_nodes: int) -> Dict[int, float]:
        if not isinstance(d, dict):
            raise TypeError(f"Intervention must be dict, got {type(d).__name__}")
        if "node" in d and "value" in d:
            idx = _Validate.node(d["node"], name_map, n_nodes)
            val = _Validate.value(d["value"])
            return {idx: val}
        out = {}
        for k, v in d.items():
            idx = _Validate.node(k, name_map, n_nodes)
            val = _Validate.value(v, name=f"value for node '{k}'")
            out[idx] = val
        return out


# ─────────────────────────────────────────────────────
#  Watch-mode state tracker
# ─────────────────────────────────────────────────────

@dataclass
class WatchCycleResult:
    """Result of a single watch() cycle."""
    cycle: int
    timestamp: float
    leakage: float
    alarms: List[Dict[str, Any]]
    revisions: List[Dict[str, Any]]
    graph_changed: bool


class _WatchState:
    """Maintains rolling statistics for the streaming alarm loop."""
    def __init__(self, graph: CausalGraph, alarm_threshold: float = 0.05, window: int = 20):
        self.graph = graph
        self.threshold = alarm_threshold
        self.window = window
        self.cycle = 0

        # Per-edge leakage buffers
        N = graph.n_nodes
        self._edge_buffers: Dict[tuple, deque] = {}
        for (i, j) in graph.edges:
            self._edge_buffers[(i, j)] = deque(maxlen=window)

        # Global leakage history
        self.leakage_history: deque = deque(maxlen=500)

        # Revision audit trail (no mocks — real RevisionEvent mirrors)
        self.revision_log: List[Dict[str, Any]] = []

    def compute_edge_leakage(self, states: np.ndarray) -> Dict[tuple, float]:
        """
        Compute per-edge causal leakage from current node states.

        Leakage(i->j) = |x_j - w_ij * x_i| / max(|x_j|, 1e-8)

        This measures how well edge (i->j) explains the child's state
        from the parent's state, weighted by the edge strength.
        Non-zero values indicate structural mismatch.
        """
        leakages = {}
        for (i, j) in self.graph.edges:
            w = self.graph.adj[i, j]
            predicted_j = w * states[i]
            residual = abs(states[j] - predicted_j)
            normalizer = max(abs(states[j]), 1e-8)
            leakages[(i, j)] = float(residual / normalizer)
        return leakages

    def step(self, states: np.ndarray, ie: InterventionEngine) -> WatchCycleResult:
        """Execute one full watch cycle with real leakage math."""
        self.cycle += 1
        ts = time.time()

        # 1. Compute real leakage from structural equations
        edge_leakages = self.compute_edge_leakage(states)
        mean_leakage = float(np.mean(list(edge_leakages.values()))) if edge_leakages else 0.0
        self.leakage_history.append(mean_leakage)

        # Update per-edge buffers
        for edge, val in edge_leakages.items():
            if edge in self._edge_buffers:
                self._edge_buffers[edge].append(val)

        # 2. Fire alarms on edges exceeding threshold (rolling mean)
        alarms = []
        for edge, buf in self._edge_buffers.items():
            if len(buf) < 3:
                continue
            rolling_mean = float(np.mean(buf))
            if rolling_mean > self.threshold:
                cycles_above = sum(1 for v in buf if v > self.threshold)
                # Artifact filter: skip if spike is < 3 cycles old
                if cycles_above < 3:
                    continue
                alarms.append({
                    "edge": edge,
                    "edge_names": (self.graph.node_name(edge[0]), self.graph.node_name(edge[1])),
                    "leakage": round(rolling_mean, 6),
                    "cycles_above_threshold": cycles_above,
                    "severity": "critical" if rolling_mean > 3 * self.threshold else "warning"
                })

        # 3. Auto-revise: propose graph surgery for critical alarms
        revisions = []
        graph_changed = False
        for alarm in alarms:
            if alarm["severity"] != "critical":
                continue
            src, tgt = alarm["edge"]

            # Real intervention simulation to test if removing the edge helps
            states_pre = states.copy()
            # Simulate severing: set edge weight to 0, re-propagate
            original_w = self.graph.adj[src, tgt]
            self.graph.adj[src, tgt] = 0.0

            post_leakages = self.compute_edge_leakage(states)
            post_mean = float(np.mean(list(post_leakages.values()))) if post_leakages else 0.0

            leakage_delta = mean_leakage - post_mean

            if leakage_delta > 0:
                # Surgery helps — accept
                # Confidence = fraction of leakage explained by this edge
                confidence = min(0.99, leakage_delta / max(mean_leakage, 1e-8))
                revisions.append({
                    "cycle": self.cycle,
                    "action": "REMOVE",
                    "edge": alarm["edge"],
                    "edge_names": alarm["edge_names"],
                    "leakage_before": round(mean_leakage, 6),
                    "leakage_after": round(post_mean, 6),
                    "leakage_reduction": round(leakage_delta, 6),
                    "confidence": round(confidence, 4),
                    "rationale": f"Removing {alarm['edge_names'][0]}->{alarm['edge_names'][1]} reduces leakage by {leakage_delta:.4f}"
                })
                graph_changed = True
                self.revision_log.append(revisions[-1])
                # Keep edge removed
            else:
                # Surgery does not help — restore
                self.graph.adj[src, tgt] = original_w

        return WatchCycleResult(
            cycle=self.cycle,
            timestamp=ts,
            leakage=mean_leakage,
            alarms=alarms,
            revisions=revisions,
            graph_changed=graph_changed
        )


# ─────────────────────────────────────────────────────
#  Fleet recurrence memory for predict_next_change()
# ─────────────────────────────────────────────────────

class _FleetRecurrenceMemory:
    """
    Tracks historical motif patterns to predict the next likely structural change.
    Uses frequency-weighted recurrence analysis (not a neural network).
    """
    def __init__(self):
        self.motif_history: List[Dict[str, Any]] = []
        self.edge_change_history: List[Dict[str, Any]] = []

    def record_change(self, cycle: int, edge: tuple, action: str,
                      leakage_before: float, confidence: float):
        self.edge_change_history.append({
            "cycle": cycle,
            "edge": edge,
            "action": action,
            "leakage_before": leakage_before,
            "confidence": confidence
        })

    def predict(self, graph: CausalGraph, current_states: np.ndarray,
                current_leakages: Dict[tuple, float]) -> Dict[str, Any]:
        """Predict the next most likely structural change using real statistics."""
        if not self.edge_change_history:
            return {
                "predicted_edge": None,
                "confidence": 0.0,
                "reasoning": "No prior structural changes recorded. Insufficient data for prediction.",
                "historical_support": 0
            }

        # Count how often each edge has been revised
        from collections import Counter
        edge_counts = Counter(e["edge"] for e in self.edge_change_history)
        total_revisions = len(self.edge_change_history)

        # Score each edge by: recurrence frequency * current leakage
        scored = []
        for edge, count in edge_counts.items():
            recurrence_freq = count / total_revisions
            current_leak = current_leakages.get(edge, 0.0)
            # Combined score: edges that recur often AND currently show leakage
            score = recurrence_freq * 0.6 + min(current_leak / 0.1, 1.0) * 0.4
            scored.append((edge, score, count, recurrence_freq, current_leak))

        # Also consider edges that have never changed but show high leakage now
        for edge, leak in current_leakages.items():
            if edge not in edge_counts and leak > 0.05:
                score = leak / 0.1 * 0.3  # Lower weight for unseen edges
                scored.append((edge, score, 0, 0.0, leak))

        if not scored:
            return {
                "predicted_edge": None,
                "confidence": 0.0,
                "reasoning": "No predictable pattern detected in revision history.",
                "historical_support": 0
            }

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0]
        edge, score, count, freq, leak = best

        # Confidence is bounded by the statistical support
        confidence = min(0.95, score * min(count + 1, 10) / 10)

        return {
            "predicted_edge": edge,
            "predicted_edge_names": (graph.node_name(edge[0]), graph.node_name(edge[1])),
            "confidence": round(float(confidence), 4),
            "recurrence_frequency": round(float(freq), 4),
            "current_leakage": round(float(leak), 6),
            "historical_support": int(count),
            "reasoning": (f"Edge {graph.node_name(edge[0])}->{graph.node_name(edge[1])} "
                          f"has been revised {count} time(s) ({freq*100:.0f}% of all revisions) "
                          f"and currently shows leakage {leak:.4f}.")
        }


# ─────────────────────────────────────────────────────
#  Main SDK classes
# ─────────────────────────────────────────────────────

class CausalNerve:
    """Main SDK entry point. Every public method delegates to a real mathematical engine."""
    
    @classmethod
    def from_preset(cls, domain_name: str) -> 'CausalNerve':
        PluginRegistry.auto_discover()
        domain = PluginRegistry.get_domain(domain_name)
        if not domain:
            raise ValueError(f"Domain '{domain_name}' not found.")
        return cls(domain=domain)

    def __init__(self, domain: DomainPlugin = None, nodes: int = None, state_dim: int = None):
        if domain is None and nodes is not None:
            class DummyDomain:
                def get_nodes(self): return {i: {"name": str(i)} for i in range(nodes)}
                def get_default_edges(self): return []
            domain = DummyDomain()
            
        self.domain = domain
        self.nodes = domain.get_nodes()
        self.edges = domain.get_default_edges()

        # Build name resolution maps
        self._node_labels = {idx: info["name"] for idx, info in self.nodes.items()}
        self._name_to_idx: Dict[str, int] = {}
        for idx, info in self.nodes.items():
            self._name_to_idx[info["name"]] = idx
            if "short" in info:
                self._name_to_idx[info["short"]] = idx

        # Build the mathematical causal graph
        self.graph = CausalGraph(
            n_nodes=len(self.nodes),
            edges=self.edges,
            node_labels=self._node_labels
        )

        # Reasoning engines (all perform real math)
        self._ie = InterventionEngine()
        self._cf = CounterfactualEngine(self._ie)
        self._tracer = CausalTracer(temporal_decay=0.85)
        self._narrator = CausalNarrator()

        # Watch-mode engine
        self._watch = _WatchState(self.graph)

        # Fleet recurrence memory
        self._fleet_memory = _FleetRecurrenceMemory()

        # Current node state vector (nominal baseline)
        self._states = np.full(len(self.nodes), 0.5)

    # ─── Validation ──────────────────────────────────

    def _resolve_node(self, node: Any) -> int:
        return _Validate.node(node, self._name_to_idx, self.graph.n_nodes)

    def _check_graph_consistency(self):
        """Verify the adjacency matrix is self-consistent."""
        adj = self.graph.adj
        if adj.shape[0] != adj.shape[1]:
            raise RuntimeError("Adjacency matrix is not square")
        if adj.shape[0] != self.graph.n_nodes:
            raise RuntimeError(
                f"Adjacency dimensions {adj.shape} != declared n_nodes {self.graph.n_nodes}"
            )
        # Check for NaN / Inf in weights
        if not np.all(np.isfinite(adj)):
            raise RuntimeError("Adjacency matrix contains NaN or Inf values")

    # ─── fit() ───────────────────────────────────────
    
    def fit(self, history: np.ndarray) -> None:
        """
        Fit the causal graph edge weights to historical baseline data.
        
        Args:
            history (np.ndarray): Historical baseline telemetry (T, N).
            
        Notes:
            Currently computes correlation-based weight initialization 
            constrained by the structural priors defined in the domain preset.
        """
        if not isinstance(history, np.ndarray):
            history = np.asarray(history, dtype=float)
            
        if history.ndim != 2 or history.shape[1] != self.graph.n_nodes:
            raise ValueError(f"History must be shape (T, {self.graph.n_nodes})")
            
        corr = np.corrcoef(history.T)
        for (i, j) in self.graph.edges:
            c = float(abs(corr[i, j]))
            self.graph.adj[i, j] = c if not np.isnan(c) else 0.1
        self._check_graph_consistency()

    # ─── why() -> CausalTracer ───────────────────────

    def why(self, node: Any = None, target: Any = None) -> dict:
        """
        Root-cause analysis via weighted backward traversal.

        Traces through the DAG using influence accumulation with temporal
        decay to rank contributing causal chains. Confidence is derived
        from the concentration of influence in the top chain.
        """
        target_node = target if target is not None else node
        if target_node is None:
            raise ValueError("Must provide either 'node' or 'target' to why()")
        idx = self._resolve_node(target_node)

        result = self._tracer.trace(
            self.graph, idx,
            node_states=self._states,
            top_k=5
        )

        chains = []
        for chain in result.ranked_causes:
            chains.append({
                "path": chain.path_names,
                "influence": round(chain.influence_score, 4),
                "contribution_pct": round(chain.contribution_pct * 100, 1),
                "depth": chain.depth
            })

        return {
            "target": result.target_name,
            "parents": [self.graph.node_name(p) for p in self.graph.parents(idx)],
            "confidence": round(result.confidence, 4),
            "ranked_chains": chains,
            "earliest_precursor": self.graph.node_name(result.earliest_precursor),
            "precursor_latency_cycles": result.precursor_latency_cycles,
            "contribution_percentages": {
                self.graph.node_name(k): round(v * 100, 1)
                for k, v in result.contribution_percentages.items()
            },
            "explanation": self._narrator.explain_root_cause(
                result, mode="engineering", node_labels=self._node_labels
            )
        }

    # ─── what_if() -> CounterfactualEngine ───────────

    def what_if(self, node: Union[int, str], value: float) -> Dict[str, Any]:
        """
        Counterfactual query via dual-world structural simulation.

        Runs 50-step factual and intervened rollouts using structural
        equations derived from the DAG. Confidence = 1 - 1/(1+D_cum).
        
        Args:
            node (Union[int, str]): The target node to intervene on.
            value (float): The clamped intervention value.
            
        Returns:
            Dict[str, Any]: The counterfactual divergence results.
        """
        idx = self._resolve_node(node)
        value = _Validate.value(value, "intervention value")

        result = self._cf.simulate(
            self.graph,
            intervention={idx: value},
            initial_states=self._states.copy(),
            horizon=50
        )

        cum_div = result.cumulative_divergence
        confidence = round(1.0 - 1.0 / (1.0 + cum_div), 4)

        return {
            "intervention": {self.graph.node_name(idx): value},
            "confidence": confidence,
            "cumulative_divergence": round(cum_div, 4),
            "leakage_reduction": round(result.leakage_reduction, 6),
            "affected_nodes": [self.graph.node_name(n) for n in result.affected_nodes],
            "unaffected_nodes": [self.graph.node_name(n) for n in result.unaffected_nodes],
            "peak_divergence": round(float(np.max(result.divergence)), 4),
            "divergence_curve": result.divergence.tolist(),
            "explanation": self._narrator.explain_intervention(
                result, {idx: value}, mode="engineering", node_labels=self._node_labels
            )
        }

    # ─── do() -> InterventionEngine ──────────────────

    def do(self, node: Union[int, str], value: float) -> Dict[str, Any]:
        """
        Pearl do(X=x): sever, clamp, propagate, verify isolation.

        Mutates the internal state vector. Returns full propagation trace.
        
        Args:
            node (Union[int, str]): The target node for the do-operator.
            value (float): The forced value for the node.
            
        Returns:
            Dict[str, Any]: The propagation results and isolation verification.
        """
        idx = self._resolve_node(node)
        value = _Validate.value(value, "intervention value")
        self._check_graph_consistency()

        result = self._ie.do(self.graph, self._states, idx, value)

        # Commit new state
        self._states = result.post_intervention_states.copy()

        return {
            "status": "success",
            "node": self.graph.node_name(idx),
            "old_value": round(float(result.pre_intervention_states[idx]), 4),
            "new_value": round(float(value), 4),
            "descendants_affected": [
                self.graph.node_name(n) for n in result.descendants_affected
            ],
            "isolation_verified": result.isolation_report.is_isolated,
            "isolation_violations": [
                self.graph.node_name(v) for v in result.isolation_report.violations
            ],
            "propagation_log": result.propagation_log,
            "post_states": {
                self.graph.node_name(i): round(float(v), 4)
                for i, v in enumerate(result.post_intervention_states)
            }
        }

    # ─── rollout() -> CounterfactualEngine ───────────

    def rollout(self, intervention: Optional[Dict[Union[int, str], float]] = None, horizon: int = 50, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Full dual-world rollout with trajectory and divergence tracking.
        
        Args:
            intervention (Dict[Union[int, str], float], optional): Node to value mapping for the intervention.
            horizon (int): The number of steps to simulate. Defaults to 50.
            steps (int, optional): Deprecated alias for horizon.
            
        Returns:
            Dict[str, Any]: Rollout simulation results.
        """
        if steps is not None:
            horizon = steps
            
        if intervention is None:
            intervention = {}
            
        int_map = _Validate.intervention_dict(
            intervention, self._name_to_idx, self.graph.n_nodes
        )

        result = self._cf.simulate(
            self.graph,
            intervention=int_map,
            initial_states=self._states.copy(),
            horizon=horizon
        )

        return {
            "divergence": round(result.cumulative_divergence, 4),
            "leakage_reduction": round(result.leakage_reduction, 6),
            "peak_divergence": round(float(np.max(result.divergence)), 4),
            "affected_nodes": [self.graph.node_name(n) for n in result.affected_nodes],
            "unaffected_nodes": [self.graph.node_name(n) for n in result.unaffected_nodes],
            "intervention_value_score": round(
                self._cf.intervention_value_score(result), 4
            ),
            "world_0_final": {
                self.graph.node_name(i): round(float(v), 4)
                for i, v in enumerate(result.world_0_trajectory[-1])
            },
            "world_1_final": {
                self.graph.node_name(i): round(float(v), 4)
                for i, v in enumerate(result.world_1_trajectory[-1])
            },
            "world_0_trajectory": result.world_0_trajectory,
            "world_1_trajectory": result.world_1_trajectory,
        }

    def run_counterfactual(self, intervention: Dict[Union[int, str], float]) -> Dict[str, Any]:
        """
        [DEPRECATED] Use `rollout()` instead.
        """
        warnings.warn(
            "run_counterfactual() is deprecated and will be removed in 2.0.0. Use rollout() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.rollout(intervention)

    # ─── step() / watch() -> _WatchState streaming ───

    def step(self, telemetry: np.ndarray) -> WatchCycleResult:
        """
        Alias for watch(), processes one step of telemetry without callbacks.
        
        Args:
            telemetry (np.ndarray): (N,) array of current sensor values per node.
            
        Returns:
            WatchCycleResult: The results of the watch cycle.
        """
        telemetry = np.asarray(telemetry, dtype=float)
        self._states = telemetry.copy()
        return self._watch.step(self._states, self._ie)

    def watch(self, telemetry_stream,
              on_alarm: Optional[Callable] = None,
              on_revision: Optional[Callable] = None) -> Optional[WatchCycleResult]:
        """
        Monitor a streaming data source for structural divergence.
        """
        telemetry_array = np.asarray(telemetry_stream, dtype=float)
        
        if np.any(np.isnan(telemetry_array)):
            raise ValueError("NaN values detected in telemetry stream")
            
        if telemetry_array.ndim == 1:
            if telemetry_array.shape[0] != self.graph.n_nodes:
                raise ValueError(f"Telemetry array must have {self.graph.n_nodes} nodes")
            return self.step(telemetry_array)
            
        try:
            cycles = 0
            alarms_total = 0
            last_result = None
            for t in telemetry_array:
                if t.shape[0] != self.graph.n_nodes:
                    raise ValueError(f"Telemetry array must have {self.graph.n_nodes} nodes")
                last_result = self.step(t)
                cycles += 1
                alarms_total += len(last_result.alarms)
                
                if on_alarm and last_result.alarms:
                    for alarm in last_result.alarms:
                        on_alarm(alarm)
                if on_revision and last_result.revisions:
                    for revision in last_result.revisions:
                        on_revision(revision)
            return last_result
        except StopIteration:
            pass
            
        return None

    # ─── predict_next_change() -> FleetRecurrenceMemory

    def predict_next_change(self) -> Dict[str, Any]:
        """
        Predict the next most likely structural change using
        frequency-weighted recurrence analysis over the revision history.

        Confidence is derived from:
          - Historical edge revision frequency (60% weight)
          - Current edge leakage magnitude (40% weight)
          - Statistical support (number of prior observations)
          
        Returns:
            Dict[str, Any]: Prediction analysis dictionary.
        """
        current_leakages = self._watch.compute_edge_leakage(self._states)
        return self._fleet_memory.predict(
            self.graph, self._states, current_leakages
        )

    # ─── IO & Exports ────────────────────────────────
    
    def export_report(self, filepath: str) -> None:
        """
        Export the current internal state and history to a JSON report.
        
        Args:
            filepath (str): The target file path.
        """
        report = {
            "n_nodes": self.graph.n_nodes,
            "edges": self.graph.edges,
            "current_states": self._states.tolist(),
            "revisions": self._watch.revision_log,
            "cycle": self._watch.cycle
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    def plot_graph(self, filepath: str = "graph.png") -> None:
        """
        Generate an SVG visualization of the current causal graph state.
        
        Args:
            filepath (str): The target file path (SVG string written directly, or `.svg` extension).
        """
        svg = self._render_graph_svg()
        actual_path = filepath.replace(".png", ".svg") if filepath.endswith(".png") else filepath
        with open(actual_path, "w", encoding="utf-8") as f:
            f.write(svg)
        if actual_path != filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(svg)

    def _render_graph_svg(self) -> str:
        N = self.graph.n_nodes
        cx, cy, r = 300, 200, 150
        positions = {}
        for i in range(N):
            angle = 2 * np.pi * i / N - np.pi / 2
            positions[i] = (cx + r * np.cos(angle), cy + r * np.sin(angle))

        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400" viewBox="0 0 600 400">']
        lines.append('<rect width="600" height="400" fill="#0D1B2A"/>')

        for (src, dst) in self.graph.edges:
            w = self.graph.adj[src, dst]
            if w < 1e-6:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            lines.append(
                f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                f'stroke="#B8860B" stroke-width="{max(1, w * 3):.1f}" opacity="0.7"/>'
            )

        for i in range(N):
            x, y = positions[i]
            s = self._states[i]
            color = "#22C55E" if s < 0.6 else ("#F59E0B" if s < 0.8 else "#DC2626")
            lines.append(
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" '
                f'fill="{color}22" stroke="{color}" stroke-width="2"/>'
            )
            lines.append(
                f'<text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle" '
                f'fill="#E2E8F0" font-size="8" font-weight="bold">'
                f'{self.graph.node_name(i)}</text>'
            )

        lines.append('</svg>')
        return '\n'.join(lines)

    def plot(self):
        raise ImportError("nerve.plot() requires causalnerve-observe. \nInstall with: pip install causalnerve-observe")
        
    def dashboard(self):
        raise ImportError("nerve.dashboard() requires causalnerve-observe. \nInstall with: pip install causalnerve-observe")

    def structural_health(self):
        """Returns the current structural health status."""
        alarms_active = []
        if self._watch.leakage_history and self._watch.leakage_history[-1] > self._watch.threshold:
            for (u, v), buf in self._watch._edge_buffers.items():
                if len(buf) > 0 and np.mean(buf) > self._watch.threshold:
                    if v not in alarms_active:
                        alarms_active.append(v)
                        
        status = "degraded" if alarms_active else "stable"
        class HealthStatus:
            def __init__(self, s, a):
                self.status, self.alarms_active = s, a
        return HealthStatus(status, alarms_active)
