"""
causalnerve.adaptation.alarm_scheduler
=================================
Implements neural-inspired refractory periods.
Prevents the same edge or local neighborhood from spamming alarms.
"""

from typing import Tuple, Dict

class RefractoryScheduler:
    """
    Manages edge-level cooldowns to ensure alarms are distinct events.
    """
    def __init__(self, base_cooldown_cycles: int = 50):
        self.base_cooldown = base_cooldown_cycles
        self.last_alarm_cycle: Dict[Tuple[int, int], int] = {}
        self.node_last_alarm_cycle: Dict[int, int] = {}
        
    def is_cooling_down(self, edge: Tuple[int, int], current_cycle: int) -> bool:
        """Check if an edge or its source node is currently within its refractory period."""
        u, v = edge
        
        # Check node-level cooldown
        if u in self.node_last_alarm_cycle:
            if current_cycle - self.node_last_alarm_cycle[u] < self.base_cooldown:
                return True
                
        # Check edge-level cooldown
        if edge in self.last_alarm_cycle:
            if current_cycle - self.last_alarm_cycle[edge] < self.base_cooldown:
                return True
                
        return False
        
    def apply_cooldown(self, edge: Tuple[int, int], current_cycle: int):
        """Mark an edge and its source node as having fired, initiating cooldown."""
        u, v = edge
        self.last_alarm_cycle[edge] = current_cycle
        self.node_last_alarm_cycle[u] = current_cycle
