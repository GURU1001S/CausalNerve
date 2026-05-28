import gzip
import json
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class ReplayFrame:
    """A complete snapshot of the system state at a specific cycle."""
    cycle: int
    timestamp: float
    telemetry: Dict[str, float]
    graph_state: Dict[str, Any]
    active_alarms: List[Dict[str, Any]] = field(default_factory=list)
    active_motifs: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)

class ReplayRecorder:
    """Records full causal state evolution for deterministic playback."""
    
    def __init__(self):
        self.frames: List[ReplayFrame] = []
        self._lock = threading.Lock()
        
    def record_frame(self, frame: ReplayFrame):
        with self._lock:
            # Keep sorted by cycle
            if not self.frames or frame.cycle > self.frames[-1].cycle:
                self.frames.append(frame)
            else:
                self.frames.append(frame)
                self.frames.sort(key=lambda f: f.cycle)

    def export_session(self, filepath: str):
        """Save the replay session as a compressed gzip JSON file."""
        with self._lock:
            data = [f.to_dict() for f in self.frames]
            
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
            
    def load_session(self, filepath: str):
        """Load a compressed replay session."""
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            
        with self._lock:
            self.frames = [ReplayFrame(**d) for d in data]
            self.frames.sort(key=lambda f: f.cycle)

class ReplayTimeline:
    """
    Manages interactive playback state, allowing pause, rewind, 
    fast-forward, and frame stepping.
    """
    def __init__(self, recorder: ReplayRecorder):
        self.recorder = recorder
        self.current_index = 0
        self.is_playing = False
        self.speed_multiplier = 1.0
        self._lock = threading.Lock()
        
    def play(self):
        with self._lock:
            self.is_playing = True
            
    def pause(self):
        with self._lock:
            self.is_playing = False
            
    def set_speed(self, speed: float):
        with self._lock:
            self.speed_multiplier = speed
            
    def step_forward(self) -> Optional[ReplayFrame]:
        with self._lock:
            if self.current_index < len(self.recorder.frames) - 1:
                self.current_index += 1
            return self.get_current_frame()
            
    def step_backward(self) -> Optional[ReplayFrame]:
        with self._lock:
            if self.current_index > 0:
                self.current_index -= 1
            return self.get_current_frame()
            
    def seek(self, target_cycle: int):
        with self._lock:
            # Binary search or simple linear scan to find closest cycle
            closest_idx = 0
            min_diff = float('inf')
            for i, f in enumerate(self.recorder.frames):
                diff = abs(f.cycle - target_cycle)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            self.current_index = closest_idx
            
    def get_current_frame(self) -> Optional[ReplayFrame]:
        if not self.recorder.frames:
            return None
        return self.recorder.frames[self.current_index]

    def get_timeline_metadata(self) -> Dict[str, Any]:
        """Provides metadata for rendering the scrubber UI."""
        frames = self.recorder.frames
        if not frames:
            return {"min_cycle": 0, "max_cycle": 0, "current_cycle": 0, "events": []}
            
        events = []
        for i, f in enumerate(frames):
            if f.interventions:
                events.append({"cycle": f.cycle, "type": "intervention"})
            if f.active_alarms:
                events.append({"cycle": f.cycle, "type": "alarm"})
                
        return {
            "min_cycle": frames[0].cycle,
            "max_cycle": frames[-1].cycle,
            "current_cycle": frames[self.current_index].cycle,
            "events": events
        }
