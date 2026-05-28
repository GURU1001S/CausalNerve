import numpy as np
from typing import Iterator, Tuple

class SyntheticStreamGenerator:
    """
    Streaming synthetic data generator for demos and testing.
    Generates data on-the-fly — no file loading needed.
    
    This is what the README quickstart uses.
    Zero external dependencies.
    """
    
    @staticmethod
    def stable(n_nodes: int = 6, n_cycles: int = 500, seed: int = 42) -> Iterator[np.ndarray]:
        """
        Yields observations one cycle at a time.
        Suitable for nerve.watch(stream) usage.
        """
        np.random.seed(seed)
        state = np.zeros(n_nodes)
        
        adj = np.zeros((n_nodes, n_nodes))
        edges = [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5)]
        for (u, v) in edges:
            adj[u, v] = 0.5
            
        for _ in range(n_cycles):
            noise = np.random.normal(0, 0.1, n_nodes)
            state = adj.T @ state + noise
            yield state.copy()
            
    @staticmethod
    def with_drift(n_nodes: int = 6, n_cycles: int = 500, drift_at: int = 200, new_edge: Tuple[int, int] = (4, 2), seed: int = 42) -> Iterator[np.ndarray]:
        """
        Stable until cycle drift_at, then adds new_edge.
        Used for the flagship demo and quickstart.
        """
        np.random.seed(seed)
        state = np.zeros(n_nodes)
        
        adj = np.zeros((n_nodes, n_nodes))
        edges = [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5)]
        for (u, v) in edges:
            adj[u, v] = 0.6
            
        cycle = 0
        while cycle < n_cycles:
            if cycle == drift_at:
                adj[new_edge[0], new_edge[1]] = 0.7
                
            noise = np.random.normal(0, 0.1, n_nodes)
            state = adj.T @ state + noise
            yield state.copy()
            cycle += 1
