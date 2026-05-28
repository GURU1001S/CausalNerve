"""
causalnerve.interventions.counterfactual
=================================
Dual-world simulation engine for causal counterfactual analysis.

Runs two parallel trajectories:
  World 0 (factual):       natural system evolution
  World 1 (counterfactual): system under do(X=x) intervention

Computes divergence D(t) = ||X_factual(t) - X_intervened(t)||
at every timestep for forensic inspection.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .intervention import CausalGraph, InterventionEngine

@dataclass
class CounterfactualResult:
    """Complete result of a dual-world counterfactual simulation."""
    world_0_trajectory: np.ndarray   # (horizon, N) factual states
    world_1_trajectory: np.ndarray   # (horizon, N) intervened states
    divergence: np.ndarray           # (horizon,) per-step L2 divergence
    per_node_divergence: np.ndarray  # (horizon, N) per-node divergence
    affected_nodes: List[int]
    unaffected_nodes: List[int]
    cumulative_divergence: float
    leakage_reduction: float
    intervention_detail: Dict

    def to_dict(self):
        return {
            "world_0_trajectory": self.world_0_trajectory.tolist(),
            "world_1_trajectory": self.world_1_trajectory.tolist(),
            "divergence": self.divergence.tolist(),
            "per_node_divergence": self.per_node_divergence.tolist(),
            "affected_nodes": self.affected_nodes,
            "unaffected_nodes": self.unaffected_nodes,
            "cumulative_divergence": self.cumulative_divergence,
            "leakage_reduction": self.leakage_reduction
        }


class CounterfactualEngine:
    """
    Dual-world causal simulation engine.
    
    Simulates natural system dynamics and intervention dynamics in parallel,
    tracking divergence at every timestep. Uses the InterventionEngine for
    rigorous do-calculus semantics.
    """
    def __init__(self, intervention_engine: Optional[InterventionEngine] = None):
        self.ie = intervention_engine or InterventionEngine()

    def _simulate_natural_dynamics(self, graph: CausalGraph, 
                                    initial_states: np.ndarray,
                                    horizon: int,
                                    noise_scale: float = 0.005) -> np.ndarray:
        """
        Simulate natural (factual) system evolution over `horizon` timesteps.
        
        Uses structural equations derived from the DAG adjacency matrix:
          X_i(t+1) = decay * X_i(t) + influence * sum_j(w_ji * X_j(t)) + noise
          
        This produces realistic degradation trajectories that respect
        the causal ordering of the graph.
        """
        N = graph.n_nodes
        trajectory = np.zeros((horizon, N))
        state = initial_states.copy()
        trajectory[0] = state.copy()
        
        topo = graph.topological_order()
        rng = np.random.RandomState(42)  # Deterministic for reproducibility
        
        for t in range(1, horizon):
            new_state = state.copy()
            for node in topo:
                parent_set = graph.parents(node)
                if parent_set:
                    # Structural equation: weighted parent influence + self-decay
                    parent_contrib = 0.0
                    total_w = 0.0
                    for p in parent_set:
                        w = graph.adj[p, node]
                        parent_contrib += w * state[p]
                        total_w += w
                    
                    if total_w > 0:
                        influence = parent_contrib / total_w
                        # Blend self-persistence with parent influence
                        alpha = min(total_w * 0.1, 0.3)  # Gradual causal coupling
                        new_state[node] = (1.0 - alpha) * state[node] + alpha * influence
                
                # Add small process noise for realistic dynamics
                new_state[node] += rng.normal(0, noise_scale)
                # Soft bounds to prevent divergence
                new_state[node] = np.clip(new_state[node], -5.0, 5.0)
            
            state = new_state
            trajectory[t] = state.copy()
        
        return trajectory

    def simulate(self, graph: CausalGraph,
                 intervention: Dict[int, float],
                 initial_states: Optional[np.ndarray] = None,
                 horizon: int = 50,
                 noise_scale: float = 0.005) -> CounterfactualResult:
        """
        Run dual-world counterfactual simulation.
        
        Args:
            graph: CausalGraph DAG
            intervention: {node_id: intervention_value} mapping
            initial_states: Starting state vector (N,). Random if None.
            horizon: Number of simulation timesteps
            noise_scale: Magnitude of process noise
            
        Returns:
            CounterfactualResult with full trajectory data
        """
        N = graph.n_nodes
        
        if initial_states is None:
            rng = np.random.RandomState(42)
            initial_states = rng.uniform(0.3, 0.7, size=(N,))
        
        # --- World 0: Factual rollout (no intervention) ---
        w0_traj = self._simulate_natural_dynamics(graph, initial_states, horizon, noise_scale)
        
        # --- World 1: Counterfactual rollout (with do-calculus intervention) ---
        # Apply intervention at t=0 and propagate
        combined_states = initial_states.copy()
        for int_node, int_val in intervention.items():
            result = self.ie.do(graph, combined_states, int_node, int_val)
            combined_states = result.post_intervention_states
        
        w1_traj = self._simulate_natural_dynamics(graph, combined_states, horizon, noise_scale)
        
        # At each timestep, re-clamp the intervention (persistent do)
        for t in range(horizon):
            for int_node, int_val in intervention.items():
                w1_traj[t, int_node] = int_val
        
        # --- Compute divergence metrics ---
        per_node_div = np.abs(w0_traj - w1_traj)              # (horizon, N)
        divergence = np.linalg.norm(w0_traj - w1_traj, axis=1) # (horizon,) L2 norm
        
        # Classify affected vs unaffected nodes
        total_div_per_node = np.mean(per_node_div, axis=0)     # (N,)
        int_nodes = set(intervention.keys())
        all_desc = set()
        for n in int_nodes:
            all_desc |= graph.descendants(n)
        
        affected = sorted(int_nodes | all_desc)
        unaffected = sorted(set(range(N)) - set(affected))
        
        # Leakage reduction estimate
        # Leakage = mean variance of prediction residuals
        w0_var = np.mean(np.var(w0_traj, axis=0))
        w1_var = np.mean(np.var(w1_traj, axis=0))
        leakage_reduction = float(max(0, w0_var - w1_var))
        
        return CounterfactualResult(
            world_0_trajectory=w0_traj,
            world_1_trajectory=w1_traj,
            divergence=divergence,
            per_node_divergence=per_node_div,
            affected_nodes=affected,
            unaffected_nodes=unaffected,
            cumulative_divergence=float(np.sum(divergence)),
            leakage_reduction=leakage_reduction,
            intervention_detail={
                "nodes": list(intervention.keys()),
                "values": list(intervention.values()),
                "node_names": [graph.node_name(n) for n in intervention.keys()]
            }
        )

    def divergence_curve(self, result: CounterfactualResult) -> np.ndarray:
        """D_CF(t) = ||W0(t) - W1(t)||^2, squared L2 divergence per step."""
        return result.divergence ** 2

    def intervention_value_score(self, result: CounterfactualResult) -> float:
        """Scalar summary: total causal impact of the intervention."""
        return float(np.sum(self.divergence_curve(result)))
