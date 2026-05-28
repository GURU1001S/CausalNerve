from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator, Iterator
import time
from dataclasses import dataclass, field

@dataclass
class TelemetryFrame:
    """A standardized unit of telemetry."""
    sensor_data: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    cycle: Optional[int] = None
    engine_id: str = "DEFAULT"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Basic validation for the telemetry frame."""
        if not isinstance(self.sensor_data, dict):
            return False
        if len(self.sensor_data) == 0:
            return False
        return True

class BaseTelemetryStream(ABC):
    """Abstract interface for all CausalNerve telemetry streams."""
    
    def __init__(self):
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    def poll(self, timeout: float = 0.0) -> Optional[TelemetryFrame]:
        """Synchronously poll the next frame. Returns None if empty."""
        pass

    def validate(self) -> bool:
        """Check if stream is healthy and data is formatted correctly."""
        return self.connected
        
    def stream_sync(self) -> Iterator[TelemetryFrame]:
        """Iterate over the stream synchronously."""
        while self.connected:
            frame = self.poll(timeout=0.1)
            if frame:
                yield frame

    @abstractmethod
    async def stream_async(self) -> AsyncIterator[TelemetryFrame]:
        """Iterate over the stream asynchronously."""
        # Must be implemented by subclasses
        yield
