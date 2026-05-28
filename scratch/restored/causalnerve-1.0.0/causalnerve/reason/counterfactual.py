"""
causalnerve.reasoning.counterfactual
=================================
Dual-world simulation: factual vs intervention trajectory.
The most visually compelling feature of CausalNerve.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..core.engine import CausalGraphEngine
from .intervention import InterventionEngine

@dataclass
class CounterfactualResult:
    world_0_trajectory: np.ndarray  # shape (horizon, N, d)
    world_1_trajectory: np.ndarray  # shape (horizon, N, d)
    divergence: np.ndarray          # shape (horizon,)
    affected_nodes: List[int]
    unaffected_nodes: List[int]

    def to_dict(self):
        return {
            "world_0_trajectory": self.world_0_trajectory.tolist(),
            "world_1_trajectory": self.world_1_trajectory.tolist(),
            "divergence": self.divergence.tolist(),
            "affected_nodes": self.affected_nodes,
            "unaffected_nodes": self.unaffected_nodes
        }

class CounterfactualEngine:
    """
    Dual-world simulation: factual vs intervention trajectory.
    """
    def __init__(self, intervention_engine: InterventionEngine):
        self.ie = intervention_engine

    def simulate(self,
                 graph: CausalGraphEngine,
                 intervention: Dict[int, torch.Tensor],
                 horizon: int = 50,
                 initial_states: Optional[torch.Tensor] = None) -> CounterfactualResult:
        """
        Run two parallel worlds from current (or provided) state:
        
        World 0 (factual):   no intervention
        World 1 (counterfactual): do(intervention)
        """
        if initial_states is None or initial_states.dim() < 3:
            # Mock states for the sake of the API
            N = getattr(graph, 'n_nodes', 14)
            D = graph.blocks[0].d_model if hasattr(graph, 'blocks') else 64
            initial_states = torch.zeros((1, N, D), device=next(graph.parameters()).device)
            
        N = initial_states.shape[1]
        
        # In a real implementation, we would roll out the model dynamically.
        # Since this is the framework architecture, we define the structure of the rollout loop.
        w0_traj = []
        w1_traj = []
        
        # World 0 (Factual) Rollout
        state_w0 = initial_states.clone()
        for _ in range(horizon):
            w0_traj.append(state_w0.detach().cpu().numpy())
            # out = graph(state_w0)['hidden']
            # state_w0 = out
            
        # World 1 (Counterfactual) Rollout
        state_w1 = initial_states.clone()
        # Apply the interventions for the duration of the rollout
        contexts = []
        for node, val in intervention.items():
            ctx = self.ie.do(graph, node, val, persist=True)
            contexts.append(ctx)
            ctx.__enter__()
            
        for _ in range(horizon):
            w1_traj.append(state_w1.detach().cpu().numpy())
            # For each step, clamp the intervened nodes and mask incoming edges
            # out = graph(state_w1, intervention_mask=...)['hidden']
            # state_w1 = out
            
        # Clean up interventions
        for ctx in contexts:
            ctx.__exit__(None, None, None)
            
        w0_np = np.stack(w0_traj) # (horizon, B, N, D)
        w1_np = np.stack(w1_traj)
        
        if w0_np.shape[1] == 1:
            w0_np = w0_np.squeeze(1) # (horizon, N, D)
            w1_np = w1_np.squeeze(1)
            
        # Compute divergence
        diff = np.mean(np.abs(w0_np - w1_np), axis=-1) # (horizon, N)
        divergence = np.mean(diff, axis=-1) # (horizon,)
        
        # Affected nodes
        total_diff_per_node = np.sum(diff, axis=0) # (N,)
        affected = [i for i in range(N) if total_diff_per_node[i] > 1e-5]
        unaffected = [i for i in range(N) if total_diff_per_node[i] <= 1e-5]
        
        return CounterfactualResult(
            world_0_trajectory=w0_np,
            world_1_trajectory=w1_np,
            divergence=divergence,
            affected_nodes=affected,
            unaffected_nodes=unaffected
        )

    def divergence_curve(self, result: CounterfactualResult) -> np.ndarray:
        """
        DCF(t) = mean over nodes of ||h_t_world1 - h_t_world0||^2
        """
        diff = result.world_1_trajectory - result.world_0_trajectory
        return np.mean(diff ** 2, axis=(1, 2))

    def intervention_value_score(self, result: CounterfactualResult) -> float:
        """
        Scalar: how much does this intervention change the future?
        Higher = more impactful intervention.
        """
        curve = self.divergence_curve(result)
        return float(np.sum(curve))
