import time
from typing import Optional, Callable
from causalnerve.runtime.base import BaseTelemetryStream, TelemetryFrame

class ReplayEngine:
    """
    Deterministic replay engine for telemetry streams.
    Handles clock drift, missing timestamps, and variable speed scaling.
    """
    def __init__(self, stream: BaseTelemetryStream, speed_multiplier: float = 1.0):
        self.stream = stream
        self.speed_multiplier = speed_multiplier
        self.is_running = False
        self._last_frame_ts: Optional[float] = None
        self._last_real_ts: Optional[float] = None

    def run(self, callback: Callable[[TelemetryFrame], None]):
        """Runs the stream, yielding frames to the callback at the correct pace."""
        self.is_running = True
        
        if not self.stream.connected:
            self.stream.connect()
            
        while self.is_running and self.stream.connected:
            frame = self.stream.poll(timeout=0.1)
            
            if frame is None:
                continue
                
            # Handle timestamps
            if self._last_frame_ts is not None and self._last_real_ts is not None:
                # Calculate required sleep
                frame_delta = frame.timestamp - self._last_frame_ts
                
                # Handle clock drift or missing timestamps
                if frame_delta < 0:
                    frame_delta = 0.05 # default fallback
                elif frame_delta > 3600:
                    frame_delta = 0.05 # cap massive gaps
                    
                target_sleep = frame_delta / self.speed_multiplier
                
                # Account for computation time
                real_delta = time.time() - self._last_real_ts
                sleep_time = target_sleep - real_delta
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            callback(frame)
            
            self._last_frame_ts = frame.timestamp
            self._last_real_ts = time.time()

    def stop(self):
        self.is_running = False
        self.stream.disconnect()
