import json
import dataclasses
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class RuntimeGraphState:
    """
    Lightweight append-only tracker for live causal monitoring state.
    """
    current_edges: int = 0
    active_alarms: List[Dict[str, Any]] = field(default_factory=list)
    accepted_surgeries: List[Dict[str, Any]] = field(default_factory=list)
    rejected_surgeries: List[Dict[str, Any]] = field(default_factory=list)
    leakage_history: List[float] = field(default_factory=list)
    uncertainty_history: List[float] = field(default_factory=list)
    lyapunov_history: List[float] = field(default_factory=list)
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def log_alarm(self, cycle: int, alarm_details: Dict[str, Any]):
        self.active_alarms.append({"cycle": cycle, **alarm_details})
        
    def log_surgery(self, cycle: int, accepted: bool, surgery_details: Dict[str, Any]):
        details = {"cycle": cycle, **surgery_details}
        if accepted:
            self.accepted_surgeries.append(details)
        else:
            self.rejected_surgeries.append(details)
            
    def export_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))
