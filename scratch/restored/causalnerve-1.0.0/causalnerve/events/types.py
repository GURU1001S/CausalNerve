from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from causalnerve.events.core import BaseEvent, EventType

@dataclass
class RevisionEvent(BaseEvent):
    """Emitted when the causal graph structure is revised/updated."""
    added_edges: List[Tuple[int, int]] = field(default_factory=list)
    removed_edges: List[Tuple[int, int]] = field(default_factory=list)
    graph_density: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.REVISION

@dataclass
class AlarmEvent(BaseEvent):
    """Emitted when leakage or divergence crosses critical thresholds."""
    leakage_value: float = 0.0
    threshold: float = 0.05
    critical_nodes: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self.event_type = EventType.ALARM

@dataclass
class InterventionEvent(BaseEvent):
    """Emitted when an active structural graph surgery is performed."""
    target_edge: Tuple[int, int] = (0, 0)
    action: str = "ACCEPT" # ACCEPT, REJECT, HOLD
    rationale: str = ""
    lyapunov_delta: float = 0.0
    w0_leakage: float = 0.0
    w1_leakage: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.INTERVENTION

@dataclass
class PredictionEvent(BaseEvent):
    """Emitted for Remaining Useful Life (RUL) or failure cycle predictions."""
    predicted_failure_cycle: int = 0
    current_cycle: int = 0
    confidence_interval: Tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        self.event_type = EventType.PREDICTION

@dataclass
class MotifEvent(BaseEvent):
    """Emitted when a known structural motif is detected."""
    motif_fingerprint: str = ""
    similarity_score: float = 0.0
    associated_macro_state: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.MOTIF

@dataclass
class FailureBoundaryEvent(BaseEvent):
    """Emitted when the system reaches the edge of chaotic non-stationarity."""
    divergence_metric: float = 0.0
    is_oscillating: bool = False
    
    def __post_init__(self):
        self.event_type = EventType.FAILURE_BOUNDARY
