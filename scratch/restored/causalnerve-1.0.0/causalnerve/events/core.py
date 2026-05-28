import time
import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from enum import Enum

class EventType(str, Enum):
    BASE = "BASE"
    REVISION = "REVISION"
    ALARM = "ALARM"
    INTERVENTION = "INTERVENTION"
    PREDICTION = "PREDICTION"
    MOTIF = "MOTIF"
    FAILURE_BOUNDARY = "FAILURE_BOUNDARY"

@dataclass
class BaseEvent:
    """Core event class for all CausalNerve telemetry and actions."""
    engine_id: str
    source_module: str
    confidence: float = 1.0
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.BASE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a primitive dictionary, safe for JSON and msgpack."""
        d = asdict(self)
        d['event_type'] = self.event_type.value
        return d

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Deserialize from dictionary."""
        # Optional: handle conversion of base to subclass based on event_type.
        return cls(**data)
