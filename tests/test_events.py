import pytest
import json
from causalnerve.events import (
    BaseEvent, EventType, EventBus,
    AlarmEvent, InterventionEvent
)

def test_base_event_serialization():
    evt = BaseEvent(engine_id="E-001", source_module="core", confidence=0.95)
    
    # Dict
    d = evt.to_dict()
    assert d["engine_id"] == "E-001"
    assert d["event_type"] == "BASE"
    assert "timestamp" in d
    
    # JSON
    j = evt.to_json()
    assert "E-001" in j
    
def test_specialized_events():
    alarm = AlarmEvent(
        engine_id="E-002", 
        source_module="leakage_monitor", 
        leakage_value=0.08,
        critical_nodes=[4, 2]
    )
    assert alarm.event_type == EventType.ALARM
    assert alarm.leakage_value == 0.08
    assert len(alarm.critical_nodes) == 2
    
    interv = InterventionEvent(
        engine_id="E-003",
        source_module="ocgr",
        target_edge=(4, 2),
        action="ACCEPT"
    )
    assert interv.event_type == EventType.INTERVENTION
    assert interv.action == "ACCEPT"

def test_event_bus_pubsub():
    bus = EventBus()
    
    received_alarms = []
    def on_alarm(evt):
        received_alarms.append(evt)
        
    received_all = []
    def on_any(evt):
        received_all.append(evt)
        
    bus.subscribe(on_alarm, EventType.ALARM)
    bus.subscribe(on_any) # All events
    
    e1 = BaseEvent(engine_id="E-01", source_module="test")
    e2 = AlarmEvent(engine_id="E-02", source_module="test", leakage_value=0.1)
    
    bus.emit(e1)
    bus.emit(e2)
    
    assert len(received_alarms) == 1
    assert received_alarms[0].engine_id == "E-02"
    
    assert len(received_all) == 2

def test_event_bus_replay():
    bus = EventBus()
    e1 = AlarmEvent(engine_id="E-01", source_module="test", leakage_value=0.1)
    bus.emit(e1)
    
    replayed = []
    bus.replay(lambda e: replayed.append(e))
    
    assert len(replayed) == 1
    assert replayed[0].event_type == EventType.ALARM
