"""
causalnerve.adaptation.alarm_system_v2
=================================
A highly stable, distributed-systems inspired alarm system for structural causal leakage.
Implements hysteresis thresholding, temporal smoothing, and cascade suppression.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque

from .alarm_localizer import AlarmLocalizer
from .alarm_scheduler import RefractoryScheduler
from .alarm_metrics import AlarmRankingMetrics

@dataclass
class FilteredAlarmEvent:
    edge: Tuple[int, int]
    cycle: int
    magnitude: float
    rank_score: float

class StructuralAlarmSystemV2:
    """
    Prevents alarm storms (e.g., 65+ simultaneous alarms) through rigorous
    hysteresis, refractory cooldowns, and spatial aggregation.
    """
    def __init__(self, 
                 threshold_on: float = 0.08, 
                 threshold_off: float = 0.04, 
                 persistence: int = 3):
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.persistence = persistence
        
        self.localizer = AlarmLocalizer()
        self.scheduler = RefractoryScheduler()
        self.ranker = AlarmRankingMetrics()
        
        # Track persistent activation
        self.edge_history: Dict[Tuple[int, int], deque] = {}
        self.active_alarms = set()

    def _apply_hysteresis(self, edge: Tuple[int, int], leakage: float) -> bool:
        """
        Applies Schmitt-trigger style hysteresis with temporal persistence.
        """
        if edge not in self.edge_history:
            self.edge_history[edge] = deque(maxlen=self.persistence)
        
        self.edge_history[edge].append(leakage)
        
        if len(self.edge_history[edge]) < self.persistence:
            return False
            
        history = list(self.edge_history[edge])
        
        # Turn ON condition: sustained above threshold_on
        if all(val > self.threshold_on for val in history):
            self.active_alarms.add(edge)
            return True
            
        # Turn OFF condition: sustained below threshold_off
        if all(val < self.threshold_off for val in history):
            self.active_alarms.discard(edge)
            return False
            
        # Maintain current state
        return edge in self.active_alarms

    def process_leakage_matrix(self, 
                               leakage_matrix: np.ndarray, 
                               cycle: int, 
                               graph_engine: Any) -> List[FilteredAlarmEvent]:
        """
        The main processing loop. Takes raw leakage, returns filtered actionable alarms.
        """
        raw_alarms = []
        n_nodes = leakage_matrix.shape[0]
        
        # 1. Hysteresis Filtering
        for u in range(n_nodes):
            for v in range(n_nodes):
                if u != v:
                    leak = leakage_matrix[u, v]
                    if self._apply_hysteresis((u, v), leak):
                        raw_alarms.append((u, v, leak))
                        
        if not raw_alarms:
            return []

        # 2. Refractory Filtering (Cooldowns)
        cooled_alarms = [
            (u, v, leak) for (u, v, leak) in raw_alarms 
            if not self.scheduler.is_cooling_down((u, v), cycle)
        ]

        # 3. Spatial Localization & Cascade Suppression
        # Collapse downstream derivative alarms into single root alarms
        localized_alarms = self.localizer.suppress_cascades(cooled_alarms, graph_engine)
        
        # 4. Ranking and dynamic cooldown check
        scored_events = []
        for (u, v, leak) in localized_alarms:
            score = self.ranker.compute_score(u, v, leak, graph_engine)
            scored_events.append((u, v, leak, score))
            
        # Sort by priority descending
        scored_events.sort(key=lambda x: x[3], reverse=True)
        
        final_events = []
        for (u, v, leak, score) in scored_events:
            # Check cooldown dynamically within the same cycle
            if not self.scheduler.is_cooling_down((u, v), cycle):
                final_events.append(FilteredAlarmEvent(edge=(u, v), cycle=cycle, magnitude=leak, rank_score=score))
                self.scheduler.apply_cooldown((u, v), cycle)
                if len(final_events) >= 5:
                    break
        
        return final_events
