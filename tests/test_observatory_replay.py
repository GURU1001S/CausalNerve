import pytest
import os
import tempfile
from causalnerve.observatory.replay import ReplayRecorder, ReplayTimeline, ReplayFrame

def test_recorder_save_load():
    recorder = ReplayRecorder()
    
    f1 = ReplayFrame(cycle=1, timestamp=1.0, telemetry={"T30": 0.5}, graph_state={"leakage_L": 0.01})
    f2 = ReplayFrame(cycle=2, timestamp=2.0, telemetry={"T30": 0.6}, graph_state={"leakage_L": 0.02})
    
    recorder.record_frame(f1)
    recorder.record_frame(f2)
    
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
        filepath = tmp.name
        
    try:
        recorder.export_session(filepath)
        assert os.path.exists(filepath)
        
        new_recorder = ReplayRecorder()
        new_recorder.load_session(filepath)
        
        assert len(new_recorder.frames) == 2
        assert new_recorder.frames[0].cycle == 1
        assert new_recorder.frames[1].telemetry["T30"] == 0.6
    finally:
        os.remove(filepath)

def test_timeline_scrubbing():
    recorder = ReplayRecorder()
    for i in range(10):
        recorder.record_frame(ReplayFrame(cycle=i, timestamp=float(i), telemetry={}, graph_state={}))
        
    timeline = ReplayTimeline(recorder)
    
    assert timeline.get_current_frame().cycle == 0
    
    timeline.step_forward()
    assert timeline.get_current_frame().cycle == 1
    
    timeline.seek(5)
    assert timeline.get_current_frame().cycle == 5
    
    timeline.seek(100) # Past max
    assert timeline.get_current_frame().cycle == 9
    
    meta = timeline.get_timeline_metadata()
    assert meta["min_cycle"] == 0
    assert meta["max_cycle"] == 9
