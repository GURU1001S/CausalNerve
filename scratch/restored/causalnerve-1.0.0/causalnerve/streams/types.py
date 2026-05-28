import queue
import time
import random
import csv
import json
import asyncio
from typing import Dict, Any, Optional, AsyncIterator
from causalnerve.runtime.base import BaseTelemetryStream, TelemetryFrame

class BufferedStream(BaseTelemetryStream):
    """
    Handles backpressure by buffering incoming telemetry.
    Drops oldest frames if max_size is reached (shedding load to prevent freezing).
    """
    def __init__(self, max_size: int = 1000):
        super().__init__()
        self.buffer: queue.Queue = queue.Queue(maxsize=max_size)

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False
        
    def push(self, frame: TelemetryFrame) -> bool:
        if not self.connected:
            return False
        try:
            self.buffer.put_nowait(frame)
            return True
        except queue.Full:
            # Backpressure handling: drop oldest
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(frame)
            except queue.Empty:
                pass
            return False

    def poll(self, timeout: float = 0.0) -> Optional[TelemetryFrame]:
        try:
            return self.buffer.get(timeout=timeout) if timeout > 0 else self.buffer.get_nowait()
        except queue.Empty:
            return None

    async def stream_async(self) -> AsyncIterator[TelemetryFrame]:
        while self.connected:
            try:
                # In a real async loop, we'd use asyncio.Queue, 
                # but for compatibility we poll with sleep.
                frame = self.buffer.get_nowait()
                yield frame
            except queue.Empty:
                await asyncio.sleep(0.01)

class SyntheticTelemetryStream(BufferedStream):
    """Generates synthetic telemetry for testing and simulation."""
    def __init__(self, rate_hz: float = 10.0, max_size: int = 1000):
        super().__init__(max_size)
        self.rate_hz = rate_hz
        self._cycle = 0

    def connect(self) -> bool:
        super().connect()
        # In a real impl, start a background thread to push data
        return True
        
    def poll(self, timeout: float = 0.0) -> Optional[TelemetryFrame]:
        # Generate on the fly for simplicity if empty
        if self.buffer.empty():
            self._cycle += 1
            frame = TelemetryFrame(
                sensor_data={"T30": 0.5 + random.random()*0.1, "P30": 0.6},
                cycle=self._cycle,
                engine_id="SYNTH-01"
            )
            self.push(frame)
        return super().poll(timeout)

class CSVReplayStream(BufferedStream):
    """Loads historical telemetry from a CSV file."""
    def __init__(self, filepath: str, max_size: int = 1000):
        super().__init__(max_size)
        self.filepath = filepath
        self._generator = None

    def connect(self) -> bool:
        super().connect()
        def _read():
            try:
                with open(self.filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cycle = int(row.pop('cycle', 0))
                        engine_id = row.pop('engine_id', 'REPLAY')
                        ts = float(row.pop('timestamp', time.time()))
                        # Parse remaining as sensors
                        sensors = {k: float(v) for k, v in row.items()}
                        yield TelemetryFrame(sensor_data=sensors, timestamp=ts, cycle=cycle, engine_id=engine_id)
            except FileNotFoundError:
                pass
        self._generator = _read()
        return True

    def poll(self, timeout: float = 0.0) -> Optional[TelemetryFrame]:
        if not self._generator:
            return None
        try:
            return next(self._generator)
        except StopIteration:
            self.disconnect()
            return None

class LiveSocketStream(BufferedStream):
    """Stub for a live WebSocket/TCP telemetry stream."""
    pass

class KafkaTelemetryStream(BufferedStream):
    """Stub for an enterprise Kafka stream consumer."""
    pass
