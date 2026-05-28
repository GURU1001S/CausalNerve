"""
benchmarks.generators
=====================
Stochastic generation of causal graphs and Structural Vector Autoregressive (SVAR) data.
Includes Erdős-Rényi, Scale-Free (Barabási-Albert), Chains, and Hierarchical graphs.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BenchmarkGraph:
    n_nodes: int
    edges: List[Tuple[int, int]]
    adj_matrix: np.ndarray  # (N, N) weighted adjacency
    
    def simulate_step(self, current_state: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Simulate one step of linear SVAR: x_t = A^T x_{t-1} + noise"""
        noise = np.random.normal(0, noise_std, size=self.n_nodes)
        return self.adj_matrix.T @ current_state + noise

class SyntheticCausalBenchmark:
    """
    Generates synthetic causal graphs with realistic topologies.
    """
    
    @staticmethod
    def _create_from_nx(G: nx.DiGraph, weight_range: Tuple[float, float] = (0.3, 0.8)) -> BenchmarkGraph:
        n = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
        adj = np.zeros((n, n))
        edges = list(G.edges())
        for (u, v) in edges:
            sign = np.random.choice([-1, 1])
            w = sign * np.random.uniform(weight_range[0], weight_range[1])
            adj[u, v] = w
        return BenchmarkGraph(n, edges, adj)

    @staticmethod
    def erdos_renyi_dag(n_nodes: int, edge_prob: float, seed: int = 42) -> BenchmarkGraph:
        np.random.seed(seed)
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < edge_prob:
                    G.add_edge(i, j)
        return SyntheticCausalBenchmark._create_from_nx(G)
        
    @staticmethod
    def scale_free_dag(n_nodes: int, m: int = 2, seed: int = 42) -> BenchmarkGraph:
        """Barabási-Albert scale-free network, directed to form a DAG."""
        np.random.seed(seed)
        G_undirected = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        G = nx.DiGraph()
        # Direct from lower index to higher to ensure DAG
        for u, v in G_undirected.edges():
            if u < v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        return SyntheticCausalBenchmark._create_from_nx(G)

    @staticmethod
    def chain_graph(n_nodes: int, seed: int = 42) -> BenchmarkGraph:
        np.random.seed(seed)
        G = nx.DiGraph()
        for i in range(n_nodes - 1):
            G.add_edge(i, i + 1)
        return SyntheticCausalBenchmark._create_from_nx(G)

    @staticmethod
    def hierarchical_graph(n_nodes: int, branching: int = 3, seed: int = 42) -> BenchmarkGraph:
        np.random.seed(seed)
        G = nx.DiGraph()
        for i in range(1, n_nodes):
            parent = (i - 1) // branching
            G.add_edge(parent, i)
        return SyntheticCausalBenchmark._create_from_nx(G)

    @staticmethod
    def feedback_graph(n_nodes: int, n_feedback: int = 2, seed: int = 42) -> BenchmarkGraph:
        np.random.seed(seed)
        G = nx.gnp_random_graph(n_nodes, 0.2, directed=True, seed=seed)
        # Randomly select a subgraph and ensure it's a DAG, then add some back edges
        DAG = nx.DiGraph([(u,v) for (u,v) in G.edges() if u < v])
        
        edges = list(DAG.edges())
        count = 0
        while count < n_feedback:
            u, v = np.random.randint(0, n_nodes, size=2)
            if u > v and (u, v) not in edges:
                DAG.add_edge(u, v)
                count += 1
        return SyntheticCausalBenchmark._create_from_nx(DAG)
