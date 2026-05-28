import threading
import queue
from typing import Callable, Dict, List, Any
from causalnerve.events.core import BaseEvent, EventType

class EventBus:
    """
    Lightweight, thread-safe Publish/Subscribe system for streaming events 
    across the CausalNerve SDK modules.
    """
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[BaseEvent], None]]] = {}
        self._all_subscribers: List[Callable[[BaseEvent], None]] = []
        self._lock = threading.Lock()
        
        # We can also store a local history for replays
        self._history: List[BaseEvent] = []

    def subscribe(self, callback: Callable[[BaseEvent], None], event_type: EventType = None):
        """
        Subscribe a callback function to specific events, or all events if event_type is None.
        """
        with self._lock:
            if event_type is None:
                self._all_subscribers.append(callback)
            else:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                self._subscribers[event_type].append(callback)

    def emit(self, event: BaseEvent):
        """
        Publish an event to all relevant subscribers.
        """
        with self._lock:
            self._history.append(event)
            # Find specific subscribers
            specific_subs = self._subscribers.get(event.event_type, [])
            # All subscribers
            all_subs = self._all_subscribers
            
            callbacks = specific_subs + all_subs

        # Execute callbacks outside the lock to prevent deadlocks
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"[EventBus] Error executing callback for {event.event_type}: {e}")

    def replay(self, callback: Callable[[BaseEvent], None]):
        """
        Replay all historical events to a specific callback. Useful for UI initialization.
        """
        with self._lock:
            history_copy = list(self._history)
            
        for event in history_copy:
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] Error replaying event {event.event_type}: {e}")
                
    def get_history(self) -> List[BaseEvent]:
        with self._lock:
            return list(self._history)
