"""
benchmarks.drift_injector
=========================
Injects realistic structural drift over time into benchmark graphs.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import copy

from .generators import BenchmarkGraph

@dataclass
class DriftBenchmark:
    base_graph: BenchmarkGraph
    drift_type: str
    ground_truth_change_cycle: int
    final_graph: BenchmarkGraph
    changed_edges: List[Tuple[int, int]]
    
class DriftInjector:
    """
    Applies controlled, stochastic, realistic structural drift.
    """
    
    @staticmethod
    def apply_drift(base_graph: BenchmarkGraph,
                    drift_type: str,
                    drift_at_cycle: int,
                    seed: int = 42) -> DriftBenchmark:
        np.random.seed(seed)
        n = base_graph.n_nodes
        adj = base_graph.adj_matrix.copy()
        current_edges = base_graph.edges.copy()
        changed_edges = []
        
        if drift_type == "add_edge":
            possible = [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in current_edges]
            if possible:
                u, v = possible[np.random.choice(len(possible))]
                sign = np.random.choice([-1, 1])
                w = sign * np.random.uniform(0.3, 0.8)
                adj[u, v] = w
                current_edges.append((u, v))
                changed_edges.append((u, v))
                
        elif drift_type == "remove_edge":
            if current_edges:
                idx = np.random.choice(len(current_edges))
                u, v = current_edges.pop(idx)
                adj[u, v] = 0.0
                changed_edges.append((u, v))
                
        elif drift_type == "edge_weight_shift":
            if current_edges:
                idx = np.random.choice(len(current_edges))
                u, v = current_edges[idx]
                adj[u, v] *= np.random.choice([0.1, 2.5]) # weaken or strengthen significantly
                changed_edges.append((u, v))
                
        elif drift_type == "regime_shift":
            # Remove 1-2 edges, add 1-2 edges
            rem_count = min(len(current_edges), np.random.randint(1, 3))
            for _ in range(rem_count):
                idx = np.random.choice(len(current_edges))
                u, v = current_edges.pop(idx)
                adj[u, v] = 0.0
                changed_edges.append((u, v))
                
            add_count = np.random.randint(1, 3)
            possible = [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in current_edges]
            for _ in range(min(add_count, len(possible))):
                idx = np.random.choice(len(possible))
                u, v = possible.pop(idx)
                sign = np.random.choice([-1, 1])
                adj[u, v] = sign * np.random.uniform(0.3, 0.8)
                current_edges.append((u, v))
                changed_edges.append((u, v))
                
        elif drift_type == "cascading_failure":
            # A node drops all outgoing connections
            if current_edges:
                node = current_edges[np.random.choice(len(current_edges))][0]
                to_remove = [e for e in current_edges if e[0] == node]
                for (u, v) in to_remove:
                    current_edges.remove((u, v))
                    adj[u, v] = 0.0
                    changed_edges.append((u, v))
                    
        final_graph = BenchmarkGraph(n, current_edges, adj)
        
        return DriftBenchmark(
            base_graph=base_graph,
            drift_type=drift_type,
            ground_truth_change_cycle=drift_at_cycle,
            final_graph=final_graph,
            changed_edges=changed_edges
        )
