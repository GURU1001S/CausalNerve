"""
causalnerve.visualization_stub.events
======================
Real-time event orchestration layer for visual drama in CausalNerve.
Defines EventBus, CausalEvent, and VisualEventScheduler.
"""

import time
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field

@dataclass
class CausalEvent:
    event_type: str  # e.g., anomaly_detected, structural_alarm, surgery_proposed, etc.
    timestamp: float
    severity: str  # healthy, warning, critical, counterfactual, accepted, rejected
    involved_nodes: List[int]
    involved_edges: List[Tuple[int, int]] = field(default_factory=list)
    explanation: str = ""
    animation_trigger: str = ""  # pulse, flash, shockwave, fade_in, fade_out
    meta: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    """
    Central event registry for broadcasting graph and decision events.
    """
    def __init__(self):
        self._listeners: Dict[str, List[Callable[[CausalEvent], None]]] = {}
        self.event_history: List[CausalEvent] = []

    def subscribe(self, event_type: str, callback: Callable[[CausalEvent], None]):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def publish(self, event: CausalEvent):
        self.event_history.append(event)
        # Publish to specific listeners
        if event.event_type in self._listeners:
            for cb in self._listeners[event.event_type]:
                cb(event)
        # Publish to wildcard listeners
        if "*" in self._listeners:
            for cb in self._listeners["*"]:
                cb(event)

class VisualEventScheduler:
    """
    Schedules visual animations and easing over discrete time cycles.
    """
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.scheduled_actions: List[Dict[str, Any]] = []

    def schedule(self, duration_sec: float, update_fn: Callable[[float], None], on_complete: Callable[[], None] = None):
        """
        Schedules a transition/easing function.
        update_fn gets a progress float between 0.0 and 1.0.
        """
        self.scheduled_actions.append({
            "start_time": time.time(),
            "duration": duration_sec,
            "update_fn": update_fn,
            "on_complete": on_complete,
            "completed": False
        })

    def step(self):
        """
        Call this periodically in the update loop to progress animations.
        """
        now = time.time()
        for action in self.scheduled_actions:
            if action["completed"]:
                continue
            elapsed = now - action["start_time"]
            progress = min(1.0, elapsed / action["duration"])
            
            # Linear easing or simple sinusoidal easing
            eased_progress = 0.5 - 0.5 * (3.14159 * (progress + 1)).cos() if hasattr(progress, 'cos') else progress
            
            action["update_fn"](progress)
            
            if progress >= 1.0:
                action["completed"] = True
                if action["on_complete"]:
                    action["on_complete"]()
                    
        # Clean up completed actions
        self.scheduled_actions = [a for a in self.scheduled_actions if not a["completed"]]
