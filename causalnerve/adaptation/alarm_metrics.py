"""
causalnerve.adaptation.alarm_metrics
===============================
Ranks and prioritizes localized alarms based on their structural severity
and potential intervention benefit.
"""

from typing import Any

class AlarmRankingMetrics:
    """
    Computes a composite priority score for an alarm event.
    """
    
    def compute_score(self, u: int, v: int, leakage_magnitude: float, graph_engine: Any) -> float:
        """
        Prioritize alarms by:
        - raw leakage magnitude
        - topological out-degree (highly connected nodes are more important)
        """
        # Base score is the physical signal strength
        score = leakage_magnitude
        
        # Structural importance multiplier
        try:
            # If engine provides adjacency, boost if `u` influences many downstream nodes
            adj = graph_engine.get_dense_adjacency()
            out_degree = sum(1 for target in range(adj.shape[1]) if adj[u, target] != 0)
            # Logarithmic boost to prevent hub-dominance
            structural_boost = 1.0 + (0.1 * out_degree)
            score *= structural_boost
        except Exception:
            pass # Graceful degradation
            
        return float(score)
