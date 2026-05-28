"""
causalnerve.interventions.trace
========================
Root-cause analysis via weighted causal graph traversal.

Implements:
  - Weighted DFS/BFS backward traversal from anomalous nodes
  - Path attribution with influence accumulation
  - Temporal decay weighting for time-aware root-cause ranking
  - Top-k causal chain extraction
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .intervention import CausalGraph

@dataclass
class CausalChain:
    """A single causal pathway from root cause to anomaly."""
    path: List[int]
    path_names: List[str]
    influence_score: float       # Product of edge weights along path
    contribution_pct: float      # Normalized share of total influence
    depth: int

@dataclass
class TraceResult:
    """Complete root-cause analysis result."""
    target_node: int
    target_name: str
    ranked_causes: List[CausalChain]
    causal_chain: List[int]      # Best single chain
    confidence: float
    contribution_percentages: Dict[int, float]
    earliest_precursor: Optional[int]
    active_anomalies: List[int]
    precursor_latency_cycles: int


class CausalTracer:
    """
    Root-cause analysis engine using weighted backward traversal.
    
    Given an anomalous node, traces backward through the DAG to identify
    the top-k contributing causal chains, with influence scores computed
    as the product of edge weights along each path, optionally decayed
    by temporal distance.
    """
    def __init__(self, temporal_decay: float = 0.85):
        self.temporal_decay = temporal_decay

    def trace(self, graph: CausalGraph, 
              anomalous_node: int,
              node_states: Optional[np.ndarray] = None,
              top_k: int = 5) -> TraceResult:
        """
        Trace backward from anomalous_node to find root causes.
        
        Args:
            graph: CausalGraph DAG
            anomalous_node: Index of the node exhibiting anomaly
            node_states: Current node state vector (N,) for anomaly scoring
            top_k: Number of top causal chains to return
            
        Returns:
            TraceResult with ranked causal chains and contribution analysis
        """
        N = graph.n_nodes
        
        # Phase 1: Enumerate all backward paths via DFS
        all_paths = []
        self._dfs_backward(graph, anomalous_node, [anomalous_node], set(), all_paths, max_depth=8)
        
        # Phase 2: Score each path
        scored_chains = []
        for path in all_paths:
            # Path is stored root -> ... -> anomalous_node
            reversed_path = list(reversed(path))
            
            # Compute influence score = product of edge weights along path
            influence = 1.0
            for i in range(len(reversed_path) - 1):
                src = reversed_path[i]
                dst = reversed_path[i + 1]
                w = graph.adj[src, dst]
                influence *= w
            
            # Apply temporal decay: deeper paths get exponentially discounted
            depth = len(reversed_path) - 1
            decay_factor = self.temporal_decay ** depth
            influence *= decay_factor
            
            # If we have node states, boost paths passing through anomalous nodes
            if node_states is not None:
                state_boost = 1.0
                for n in reversed_path[:-1]:  # Exclude target
                    anomaly_magnitude = abs(node_states[n] - 0.5)  # Distance from nominal
                    state_boost *= (1.0 + anomaly_magnitude)
                influence *= state_boost
            
            scored_chains.append(CausalChain(
                path=reversed_path,
                path_names=[graph.node_name(n) for n in reversed_path],
                influence_score=influence,
                contribution_pct=0.0,  # Normalized later
                depth=depth
            ))
        
        # Phase 3: Normalize contribution percentages
        total_influence = sum(c.influence_score for c in scored_chains) or 1.0
        for chain in scored_chains:
            chain.contribution_pct = chain.influence_score / total_influence
        
        # Sort by influence (descending) and take top-k
        scored_chains.sort(key=lambda c: c.influence_score, reverse=True)
        top_chains = scored_chains[:top_k]
        
        # Phase 4: Aggregate per-node contribution
        node_contributions: Dict[int, float] = defaultdict(float)
        for chain in scored_chains:
            root = chain.path[0]
            node_contributions[root] += chain.contribution_pct
        
        # Normalize per-node contributions
        total_node = sum(node_contributions.values()) or 1.0
        contribution_percentages = {n: v / total_node for n, v in node_contributions.items()}
        
        # Phase 5: Identify earliest precursor (root of the strongest chain)
        best_chain = top_chains[0] if top_chains else None
        earliest_precursor = best_chain.path[0] if best_chain else anomalous_node
        causal_chain = best_chain.path if best_chain else [anomalous_node]
        
        # Confidence based on how concentrated the influence is
        if top_chains:
            top_share = top_chains[0].contribution_pct
            confidence = min(0.99, 0.5 + top_share)
        else:
            confidence = 0.3
        
        # Precursor latency estimate (depth of best chain * assumed cycle delay)
        latency = len(causal_chain) * 3
        
        return TraceResult(
            target_node=anomalous_node,
            target_name=graph.node_name(anomalous_node),
            ranked_causes=top_chains,
            causal_chain=causal_chain,
            confidence=confidence,
            contribution_percentages=contribution_percentages,
            earliest_precursor=earliest_precursor,
            active_anomalies=[anomalous_node],
            precursor_latency_cycles=latency
        )
    
    def _dfs_backward(self, graph: CausalGraph, current: int,
                       path: List[int], visited: Set[int],
                       all_paths: List[List[int]], max_depth: int):
        """Recursive backward DFS through the DAG."""
        if len(path) > max_depth:
            return
        
        parents = graph.parents(current)
        
        if not parents or (parents <= visited):
            # Leaf node or all parents already visited — record this path
            if len(path) > 1:
                all_paths.append(list(path))
            return
        
        for p in parents:
            if p not in visited:
                new_visited = visited | {p}
                new_path = path + [p]
                # Record every intermediate path too
                if len(new_path) > 1:
                    all_paths.append(list(new_path))
                self._dfs_backward(graph, p, new_path, new_visited, all_paths, max_depth)
