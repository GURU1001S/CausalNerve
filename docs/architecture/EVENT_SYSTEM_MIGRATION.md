# CausalNerve SDK: Event System Migration Plan

## Overview
CausalNerve has matured from a monolithic prototype into a modular, production-grade SDK. To support high-frequency streaming, decoupling of sub-modules, and robust replayability, we have introduced a unified **Event Bus** architecture (`causalnerve/events/`).

This document outlines the migration strategy for transitioning existing modules to the new Event System.

## Architecture

1. **`BaseEvent` & Types**: A strictly typed dataclass hierarchy encompassing all states (e.g., `AlarmEvent`, `InterventionEvent`, `MotifEvent`).
2. **`EventBus`**: A lightweight, thread-safe publish/subscribe broker.
3. **Serialization**: Native to-dict and JSON serializers designed to stream across sockets, Kafka, or file systems with zero friction.

## Migration Steps

### Phase 1: Engine Initialization
Replace direct function calls across modules with a global (or per-engine) `EventBus`.

```python
from causalnerve.events import EventBus, AlarmEvent, EventType

# Initialize bus
bus = EventBus()
```

### Phase 2: Decoupling Modules (Publishers)
Instead of the `PhysicalConstraintEngine` or `EarlyWarningEngine` calling methods directly on the Observatory or Report Generator, they simply emit events to the bus.

*Before:*
```python
def check_alarm(leakage):
    if leakage > 0.05:
        observatory.trigger_alarm(leakage)
```

*After:*
```python
def check_alarm(leakage):
    if leakage > 0.05:
        bus.emit(AlarmEvent(
            engine_id="E-LIVE",
            source_module="leakage_monitor",
            leakage_value=leakage,
            critical_nodes=[4, 2]
        ))
```

### Phase 3: Decoupling the Observatory (Subscribers)
The `causalnerve_observatory.py` backend currently polls a massive `data` dictionary. It will migrate to an event-driven subscriber model.

```python
class CausalNerveObservatory:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.bus.subscribe(self.handle_alarm, EventType.ALARM)
        self.bus.subscribe(self.handle_intervention, EventType.INTERVENTION)
        
    def handle_alarm(self, event: AlarmEvent):
        self._state["status"]["alarm_active"] = True
        self._state["leakage_L"] = event.leakage_value
```

### Phase 4: Replay & Audit Integration
Because every structural surgery and hypothesis is now typed and tracked by the `EventBus`, the replay system simply instantiates an empty pipeline and calls `bus.replay(observatory.handle_event)`.

Similarly, the `ScientificReportGenerator` subscribes to the bus and silently compiles its Markdown report in the background without injecting itself into the core loop.

## Immediate Action Items
- [ ] Refactor `causalnerve_demo.py` to instantiate `EventBus`.
- [ ] Pipe `InterventionRecord` through `InterventionEvent`.
- [ ] Migrate `CausalNerveObservatory`'s `_process` function to consume events via WebSockets or polling event queues instead of dictionary merges.
