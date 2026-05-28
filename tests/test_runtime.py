import pytest
import os
import csv
import tempfile
import time
from causalnerve.runtime import (
    TelemetryFrame,
    SyntheticTelemetryStream,
    CSVReplayStream,
    BufferedStream,
    ReplayEngine
)

def test_telemetry_frame_validation():
    frame = TelemetryFrame(sensor_data={"T30": 0.5}, cycle=1)
    assert frame.validate() is True
    
    bad_frame = TelemetryFrame(sensor_data={}, cycle=2)
    assert bad_frame.validate() is False

def test_buffered_stream_backpressure():
    stream = BufferedStream(max_size=3)
    stream.connect()
    
    stream.push(TelemetryFrame({"s": 1}, cycle=1))
    stream.push(TelemetryFrame({"s": 2}, cycle=2))
    stream.push(TelemetryFrame({"s": 3}, cycle=3))
    
    # Push 4th should drop the 1st
    success = stream.push(TelemetryFrame({"s": 4}, cycle=4))
    assert success is False # Indicates backpressure dropped a frame
    
    # Next poll should be cycle 2
    frame = stream.poll()
    assert frame is not None
    assert frame.cycle == 2

def test_synthetic_stream():
    stream = SyntheticTelemetryStream()
    stream.connect()
    
    frame = stream.poll()
    assert frame is not None
    assert frame.cycle == 1
    assert "T30" in frame.sensor_data

def test_csv_replay_stream():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as f:
        writer = csv.DictWriter(f, fieldnames=["cycle", "engine_id", "timestamp", "T30", "P30"])
        writer.writeheader()
        writer.writerow({"cycle": 1, "engine_id": "E1", "timestamp": time.time(), "T30": 0.5, "P30": 0.6})
        writer.writerow({"cycle": 2, "engine_id": "E1", "timestamp": time.time()+0.1, "T30": 0.55, "P30": 0.65})
        temp_name = f.name
        
    try:
        stream = CSVReplayStream(temp_name)
        stream.connect()
        
        f1 = stream.poll()
        assert f1.cycle == 1
        assert f1.sensor_data["T30"] == 0.5
        
        f2 = stream.poll()
        assert f2.cycle == 2
        
        f3 = stream.poll()
        assert f3 is None
    finally:
        os.remove(temp_name)

def test_replay_engine():
    stream = SyntheticTelemetryStream()
    engine = ReplayEngine(stream, speed_multiplier=100.0)
    
    received = []
    def callback(frame):
        received.append(frame)
        if len(received) >= 3:
            engine.stop()
            
    engine.run(callback)
    
    assert len(received) == 3
    assert received[0].cycle == 1
    assert received[2].cycle == 3
