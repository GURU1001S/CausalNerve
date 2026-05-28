import numpy as np

def connectivity_entropy(adj_matrix: np.ndarray) -> float:
    """Shannon entropy of the connectivity matrix, measuring structural randomness."""
    p = adj_matrix.flatten()
    p = p[p > 0]
    p = p / np.sum(p) if np.sum(p) > 0 else np.array([1.0])
    return -np.sum(p * np.log(p + 1e-9))

def graph_volatility(adj_t1: np.ndarray, adj_t2: np.ndarray) -> float:
    """Measures how much the causal graph changed between two consecutive windows."""
    return np.mean(np.abs(adj_t1 - adj_t2))

def synchronization_index(data_window: np.ndarray) -> float:
    """Phase synchronization proxy (Kuramoto order parameter analog) for the window."""
    # Simple proxy: mean absolute correlation of the analytic signal or just raw correlation
    corr = np.corrcoef(data_window.T)
    np.fill_diagonal(corr, 0)
    return np.mean(np.abs(corr))

def temporal_graph_coherence(history_adjs: list) -> float:
    """Measures if the graph topology persists or oscillates wildly over time."""
    if len(history_adjs) < 2: return 1.0
    vols = [graph_volatility(history_adjs[i], history_adjs[i-1]) for i in range(1, len(history_adjs))]
    return 1.0 / (1.0 + np.mean(vols))

def causal_motif_persistence(motifs_t1: list, motifs_t2: list) -> float:
    """Measures survival rate of functional causal motifs (e.g., chains, colliders)."""
    set1 = set(motifs_t1)
    set2 = set(motifs_t2)
    if not set1: return 0.0
    return len(set1.intersection(set2)) / len(set1)

def intervention_effectiveness(factual_trajectory: np.ndarray, 
                               intervened_trajectory: np.ndarray, 
                               target_nodes: list) -> float:
    """
    Measures how much the intervention suppressed the targeted abnormal activity 
    (e.g., seizure-like connectivity explosion or leakage spike).
    """
    diff = np.abs(factual_trajectory[:, target_nodes] - intervened_trajectory[:, target_nodes])
    return np.mean(diff)
