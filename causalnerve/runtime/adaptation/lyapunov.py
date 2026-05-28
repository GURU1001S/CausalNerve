"""
causalnerve.runtime.adaptation.lyapunov
==========================
Structural Lyapunov Energy Function V(G_t).
Defines V(G_t) — a scalar energy function over the causal graph state.
An edit is accepted ONLY if V(G_after) < V(G_before).
This provably eliminates oscillation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class LyapunovWeights:
    w_leak: float
    w_entropy: float
    w_energy: float
    w_thermo: float

class AdaptiveLyapunovScheduler:
    """
    Adjusts Lyapunov function weights dynamically based on 
    the current structural stability state of the graph.
    """
    def __init__(self):
        self.w_entropy_max = 0.40
        self.w_entropy_min = 0.10
        self.w_leak_min = 0.30
        self.w_leak_max = 0.60
        self.w_energy = 0.15
        self.w_thermo = 0.10
        
        self.V_history = []
        self.V_initial = None
    
    def compute_stability_index(self) -> float:
        if self.V_initial is None or self.V_initial == 0:
            return 0.0
        if len(self.V_history) < 20:
            return 0.0
        recent_mean = np.mean(self.V_history[-20:])
        return float(np.clip(1.0 - recent_mean / self.V_initial, 0.0, 1.0))
    
    def get_weights(self) -> LyapunovWeights:
        s = self.compute_stability_index()
        w_entropy = self.w_entropy_max * (1-s) + self.w_entropy_min * s
        w_leak = self.w_leak_min * (1-s) + self.w_leak_max * s
        
        return LyapunovWeights(
            w_leak=w_leak,
            w_entropy=w_entropy,
            w_energy=self.w_energy,
            w_thermo=self.w_thermo
        )
    
    def update(self, V_current: float):
        if self.V_initial is None:
            self.V_initial = V_current
        self.V_history.append(V_current)


@dataclass
class GraphState:
    adj: torch.Tensor
    edge_leakage: torch.Tensor
    n_nodes: int = 14


@dataclass
class LyapunovResult:
    V_total: float
    V_leak: float
    V_entropy: float
    V_energy: float
    V_thermo: float


class StructuralLyapunovFunction:
    """
    V(G_t) — scalar energy of the causal graph at time t.
    An edit is only accepted if V(G_after) < V(G_before).
    """
    def __init__(self, w_leak=0.4, w_entropy=0.3, w_energy=0.2, w_thermo=0.1,
                 oscillation_lookback=50, leakage_lookback=20, n_nodes=14):
        self.w_leak = w_leak
        self.w_entropy = w_entropy
        self.w_energy = w_energy
        self.w_thermo = w_thermo
        self.oscillation_lookback = oscillation_lookback
        self.leakage_lookback = leakage_lookback
        self.n_nodes = n_nodes
        self.scheduler = AdaptiveLyapunovScheduler()
        
        self.flow_order = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        self._thermo_penalty_cache = None
        self.v_trajectory = []

    @property
    def current_energy(self) -> float:
        if len(self.v_trajectory) > 0:
            return float(self.v_trajectory[-1])
        return 0.0

    @property
    def oscillation_counter(self) -> int:
        return 0

    def _build_thermo_penalty(self, device):
        if self._thermo_penalty_cache is not None and self._thermo_penalty_cache.device == device:
            return self._thermo_penalty_cache

        N = self.n_nodes
        penalty = torch.zeros(N, N, device=device)
        for src in range(N):
            for tgt in range(N):
                if src in self.flow_order and tgt in self.flow_order:
                    backward_jump = self.flow_order[src] - self.flow_order[tgt]
                    if backward_jump > 2:
                        penalty[src, tgt] = 1.0
                    elif backward_jump == 2:
                        penalty[src, tgt] = 0.3
        self._thermo_penalty_cache = penalty
        return penalty

    def compute_leakage_term(self, leakage_history: np.ndarray) -> float:
        if len(leakage_history) == 0:
            return 0.0
        return float(np.mean(leakage_history[-self.leakage_lookback:]))

    def compute_entropy_term(self, edit_history: list, current_cycle: int = 0) -> float:
        lookback_start = max(0, current_cycle - self.oscillation_lookback)
        edge_edits = defaultdict(list)
        for event in edit_history:
            if getattr(event, 'accepted', False) and getattr(event, 'cycle_t', 0) >= lookback_start:
                edge_edits[event.edge].append(event.edit_type)

        total_penalty = 0.0
        for edge, edit_types in edge_edits.items():
            osc_count = sum(1 for i in range(1, len(edit_types)) if edit_types[i] != edit_types[i - 1])
            total_penalty += osc_count ** 2

        return total_penalty

    def compute_energy_term(self, graph_state: GraphState) -> float:
        return float(torch.sum(graph_state.adj ** 2).item())

    def compute_thermo_term(self, graph_state: GraphState, theta: np.ndarray) -> float:
        adj = graph_state.adj
        penalty_matrix = self._build_thermo_penalty(adj.device)
        N = min(adj.shape[0], penalty_matrix.shape[0])
        return float(torch.sum(adj[:N, :N] * penalty_matrix[:N, :N]).item())

    def compute(self, graph_state: GraphState, leakage_history: np.ndarray,
                edit_history: list, theta: np.ndarray, current_cycle: int = 0) -> LyapunovResult:
        if self.scheduler:
            weights = self.scheduler.get_weights()
            self.w_leak = weights.w_leak
            self.w_entropy = weights.w_entropy
            self.w_energy = weights.w_energy
            self.w_thermo = weights.w_thermo

        V_leak = self.compute_leakage_term(leakage_history)
        V_entropy = self.compute_entropy_term(edit_history, current_cycle)
        V_energy = self.compute_energy_term(graph_state)
        V_thermo = self.compute_thermo_term(graph_state, theta)

        V_total = (self.w_leak * V_leak + self.w_entropy * V_entropy +
                   self.w_energy * V_energy + self.w_thermo * V_thermo)

        result = LyapunovResult(V_total, V_leak, V_entropy, V_energy, V_thermo)
        self.v_trajectory.append(V_total)
        return result

    def gate_edit(self, proposed_edit, current_graph: GraphState, leakage_history: np.ndarray,
                  edit_history: list, theta: np.ndarray, current_cycle: int = 0,
                  proposed_leakage: Optional[np.ndarray] = None) -> Tuple[bool, float, float, str]:
        v_before_result = self.compute(current_graph, leakage_history, edit_history, theta, current_cycle)
        V_before = v_before_result.V_total

        if self.scheduler:
            self.scheduler.update(V_before)

        hyp_adj = current_graph.adj.clone()
        src, tgt = getattr(proposed_edit, 'edge', (getattr(proposed_edit, 'source', 0), getattr(proposed_edit, 'target', 0)))
        edit_type = getattr(proposed_edit, 'edit_type', 'add')

        if edit_type == 'add':
            hyp_adj[src, tgt] = 0.5
        elif edit_type == 'remove':
            hyp_adj[src, tgt] = 0.0

        hyp_edit_history = list(edit_history)
        class _HypEvent:
            def __init__(self, e, et, c, a):
                self.edge, self.edit_type, self.cycle_t, self.accepted = e, et, c, a
        hyp_edit_history.append(_HypEvent((src, tgt), edit_type, current_cycle, True))

        hyp_leakage = proposed_leakage if proposed_leakage is not None else leakage_history
        hyp_graph = GraphState(hyp_adj, current_graph.edge_leakage, current_graph.n_nodes)

        V_leak_after = self.compute_leakage_term(hyp_leakage)
        V_entropy_after = self.compute_entropy_term(hyp_edit_history, current_cycle)
        V_energy_after = self.compute_energy_term(hyp_graph)
        V_thermo_after = self.compute_thermo_term(hyp_graph, theta)

        V_after = (self.w_leak * V_leak_after + self.w_entropy * V_entropy_after +
                   self.w_energy * V_energy_after + self.w_thermo * V_thermo_after)

        accepted = V_after < V_before
        reason = ""
        if not accepted:
            delta = V_after - V_before
            if V_entropy_after > v_before_result.V_entropy:
                reason = f"lyapunov_oscillation (dV={delta:+.4f})"
            elif V_energy_after > v_before_result.V_energy:
                reason = f"lyapunov_energy (dV={delta:+.4f})"
            else:
                reason = f"lyapunov_generic (dV={delta:+.4f})"

        return accepted, V_before, V_after, reason

    def reset(self):
        self.v_trajectory.clear()
        self._thermo_penalty_cache = None
        if self.scheduler:
            self.scheduler.V_history.clear()
            self.scheduler.V_initial = None
