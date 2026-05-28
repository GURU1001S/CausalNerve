"""
causalnerve.adaptation.alarm_localizer
=================================
Suppresses derivative alarms and cascading false positives using topological containment.
"""

from typing import List, Tuple, Any
import numpy as np

class AlarmLocalizer:
    """
    Identifies root causes of alarm cascades.
    If multiple connected edges alarm simultaneously, prioritizes the most upstream edge.
    """
    
    def suppress_cascades(self, 
                          alarms: List[Tuple[int, int, float]], 
                          graph_engine: Any) -> List[Tuple[int, int, float]]:
        """
        Takes a list of (u, v, magnitude) alarms.
        Uses graph topology to eliminate downstream derivative alarms.
        """
        if not alarms:
            return []
            
        # Get dense adjacency
        if hasattr(graph_engine, 'get_dense_adjacency'):
            adj = graph_engine.get_dense_adjacency()
        else:
            n = max(max(u, v) for u, v, _ in alarms) + 1
            adj = np.zeros((n, n))
            
        suppressed = set()
        
        # Sort alarms by magnitude descending (heuristic to start with strongest signal)
        alarms_sorted = sorted(alarms, key=lambda x: x[2], reverse=True)
        
        active_nodes = {u for u, v, _ in alarms_sorted} | {v for u, v, _ in alarms_sorted}
        
        # Find paths between alarmed edges
        # If Edge(A->B) and Edge(B->C) both alarm, suppress B->C
        for i, (u1, v1, mag1) in enumerate(alarms_sorted):
            if (u1, v1) in suppressed:
                continue
            for j, (u2, v2, mag2) in enumerate(alarms_sorted):
                if i == j or (u2, v2) in suppressed:
                    continue
                
                # Direct topological cascade: v1 == u2 (A->B -> B->C)
                if v1 == u2:
                    suppressed.add((u2, v2))
                    
                # Sibling cascade: same source node (A->B and A->C)
                # Since alarms_sorted is sorted by magnitude, this keeps the strongest
                if u1 == u2:
                    suppressed.add((u2, v2))
                    
        filtered = [alarm for alarm in alarms_sorted if (alarm[0], alarm[1]) not in suppressed]
        return filtered
