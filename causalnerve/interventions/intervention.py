"""
causalnerve.interventions.intervention
===============================
Pearl-style do-calculus as exact structural graph operations.

Implements do(X=x): sever incoming edges, clamp state, propagate downstream.
All operations are deterministic and mathematically traceable.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass, field

@dataclass
class IsolationReport:
    """Result of verifying the causal isolation guarantee."""
    is_isolated: bool
    violations: List[int]
    divergence_map: Dict[int, float]

@dataclass
class InterventionResult:
    """Complete result of a do(X=x) intervention on a causal graph."""
    pre_intervention_states: np.ndarray      # (N,) node values before
    post_intervention_states: np.ndarray     # (N,) node values after
    intervened_node: int
    intervention_value: float
    descendants_affected: List[int]
    non_descendants_unchanged: List[int]
    isolation_report: IsolationReport
    propagation_log: List[Dict]              # step-by-step causal flow


class CausalGraph:
    """
    Lightweight adjacency-based causal DAG for mathematical reasoning.
    Operates on numpy arrays for maximum portability (no torch dependency).
    """
    def __init__(self, n_nodes: int, edges: List[Tuple[int, int]], 
                 edge_weights: Optional[Dict[Tuple[int, int], float]] = None,
                 node_labels: Optional[Dict[int, str]] = None):
        self.n_nodes = n_nodes
        self.edges = list(edges)
        self.node_labels = node_labels or {i: f"X{i}" for i in range(n_nodes)}
        
        # Build weighted adjacency matrix
        self.adj = np.zeros((n_nodes, n_nodes))
        for (i, j) in self.edges:
            w = 1.0
            if edge_weights and (i, j) in edge_weights:
                w = edge_weights[(i, j)]
            self.adj[i, j] = w
            
        # Precompute structural relationships
        self._children_cache: Dict[int, Set[int]] = {}
        self._parents_cache: Dict[int, Set[int]] = {}
        self._descendants_cache: Dict[int, Set[int]] = {}
        self._ancestors_cache: Dict[int, Set[int]] = {}
        self._topo_order: Optional[List[int]] = None
        
    def children(self, node: int) -> Set[int]:
        """Direct children of a node."""
        if node not in self._children_cache:
            self._children_cache[node] = set(np.where(self.adj[node, :] > 0)[0])
        return self._children_cache[node]
    
    def parents(self, node: int) -> Set[int]:
        """Direct parents of a node."""
        if node not in self._parents_cache:
            self._parents_cache[node] = set(np.where(self.adj[:, node] > 0)[0])
        return self._parents_cache[node]
    
    def descendants(self, node: int) -> Set[int]:
        """All descendants of a node (BFS)."""
        if node not in self._descendants_cache:
            visited = set()
            queue = list(self.children(node))
            while queue:
                curr = queue.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    queue.extend(self.children(curr) - visited)
            self._descendants_cache[node] = visited
        return self._descendants_cache[node]
    
    def ancestors(self, node: int) -> Set[int]:
        """All ancestors of a node (reverse BFS)."""
        if node not in self._ancestors_cache:
            visited = set()
            queue = list(self.parents(node))
            while queue:
                curr = queue.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    queue.extend(self.parents(curr) - visited)
            self._ancestors_cache[node] = visited
        return self._ancestors_cache[node]
    
    def topological_order(self) -> List[int]:
        """Kahn's algorithm for topological sort."""
        if self._topo_order is not None:
            return self._topo_order
        in_degree = np.sum(self.adj > 0, axis=0).astype(int)
        queue = [i for i in range(self.n_nodes) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # If cycle exists, append remaining nodes
        for i in range(self.n_nodes):
            if i not in order:
                order.append(i)
        self._topo_order = order
        return order
    
    def node_name(self, idx: int) -> str:
        return self.node_labels.get(idx, f"X{idx}")


class InterventionEngine:
    """
    Pearl-style structural interventions on causal DAGs.
    
    do(X_i = x):
      1. Sever all incoming edges to X_i (remove from structural equations)
      2. Clamp X_i = x
      3. Propagate effects downstream through the graph in topological order
      4. Verify isolation: non-descendants must remain unchanged
    """
    
    def __init__(self):
        self._active_interventions: Dict[int, float] = {}
    
    def do(self, graph: CausalGraph, states: np.ndarray, 
           node: int, value: float) -> InterventionResult:
        """
        Apply do(X_node = value) to the graph states.
        
        Args:
            graph: The causal DAG
            states: Current node states, shape (N,)
            node: Index of node to intervene on
            value: Intervention value to clamp
            
        Returns:
            InterventionResult with full propagation trace
        """
        N = graph.n_nodes
        pre_states = states.copy()
        post_states = states.copy()
        propagation_log = []
        
        # Step 1: Sever incoming edges (conceptually — we skip parent contributions)
        severed_parents = list(graph.parents(node))
        propagation_log.append({
            "step": "sever",
            "node": node,
            "node_name": graph.node_name(node),
            "severed_parents": [graph.node_name(p) for p in severed_parents],
            "detail": f"Removed {len(severed_parents)} incoming edges to {graph.node_name(node)}"
        })
        
        # Step 2: Clamp the intervened node
        old_val = post_states[node]
        post_states[node] = value
        propagation_log.append({
            "step": "clamp",
            "node": node,
            "node_name": graph.node_name(node),
            "old_value": float(old_val),
            "new_value": float(value),
            "detail": f"Clamped {graph.node_name(node)}: {old_val:.4f} -> {value:.4f}"
        })
        
        # Step 3: Propagate downstream in topological order
        desc = graph.descendants(node)
        topo = graph.topological_order()
        
        # Only propagate through descendants, in topological order
        for target in topo:
            if target not in desc:
                continue
            
            # Compute new value for target from its parents' post-intervention states
            parent_set = graph.parents(target)
            if not parent_set:
                continue
                
            # Weighted sum of parent contributions
            weighted_sum = 0.0
            total_weight = 0.0
            for p in parent_set:
                w = graph.adj[p, target]
                weighted_sum += w * post_states[p]
                total_weight += w
                
            if total_weight > 0:
                # Structural equation: X_target = f(parents) 
                # Using weighted average with damping to preserve scale
                parent_influence = weighted_sum / total_weight
                # Blend: how much the node state is driven by parents vs. its own baseline
                alpha = min(total_weight, 1.0)  # Influence strength capped at 1
                new_val = alpha * parent_influence + (1.0 - alpha) * pre_states[target]
                
                delta = new_val - post_states[target]
                post_states[target] = new_val
                
                if abs(delta) > 1e-8:
                    propagation_log.append({
                        "step": "propagate",
                        "node": target,
                        "node_name": graph.node_name(target),
                        "contributing_parents": [graph.node_name(p) for p in parent_set],
                        "delta": float(delta),
                        "new_value": float(new_val),
                        "detail": f"{graph.node_name(target)}: delta={delta:+.6f} from parents {[graph.node_name(p) for p in parent_set]}"
                    })
        
        # Step 4: Isolation verification
        non_desc = set(range(N)) - desc - {node}
        divergence_map = {}
        violations = []
        for i in range(N):
            div = abs(post_states[i] - pre_states[i])
            divergence_map[i] = float(div)
            if i in non_desc and div > 1e-8:
                violations.append(i)
        
        isolation = IsolationReport(
            is_isolated=len(violations) == 0,
            violations=violations,
            divergence_map=divergence_map
        )
        
        return InterventionResult(
            pre_intervention_states=pre_states,
            post_intervention_states=post_states,
            intervened_node=node,
            intervention_value=value,
            descendants_affected=sorted(desc),
            non_descendants_unchanged=sorted(non_desc),
            isolation_report=isolation,
            propagation_log=propagation_log
        )
    
    def multi_do(self, graph: CausalGraph, states: np.ndarray,
                 interventions: Dict[int, float]) -> InterventionResult:
        """Apply multiple simultaneous interventions."""
        current_states = states.copy()
        all_logs = []
        all_desc = set()
        
        for node, value in interventions.items():
            result = self.do(graph, current_states, node, value)
            current_states = result.post_intervention_states
            all_logs.extend(result.propagation_log)
            all_desc.update(result.descendants_affected)
        
        non_desc = set(range(graph.n_nodes)) - all_desc - set(interventions.keys())
        divergence_map = {i: float(abs(current_states[i] - states[i])) for i in range(graph.n_nodes)}
        violations = [i for i in non_desc if divergence_map[i] > 1e-8]
        
        return InterventionResult(
            pre_intervention_states=states,
            post_intervention_states=current_states,
            intervened_node=list(interventions.keys())[0],
            intervention_value=list(interventions.values())[0],
            descendants_affected=sorted(all_desc),
            non_descendants_unchanged=sorted(non_desc),
            isolation_report=IsolationReport(
                is_isolated=len(violations) == 0,
                violations=violations,
                divergence_map=divergence_map
            ),
            propagation_log=all_logs
        )
