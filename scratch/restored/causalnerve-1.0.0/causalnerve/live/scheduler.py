from typing import Callable, Any, Optional
from .stream import LiveCMAPSSStream
from .runtime_state import RuntimeGraphState

class LiveMonitoringScheduler:
    """
    Pulls from LiveCMAPSSStream, feeds CausalNerve.watch(), 
    triggers adaptation loop, updates RuntimeGraphState, handles 
    alarm cooldown timing, handles asynchronous graph revisions safely.
    """
    def __init__(self, nerve, stream: LiveCMAPSSStream, cooldown_cycles: int = 5):
        self.nerve = nerve
        self.stream = stream
        self.state = RuntimeGraphState()
        self.cooldown_cycles = cooldown_cycles
        self._last_alarm_cycle = -self.cooldown_cycles - 1
        self.base_threshold = nerve.alarm_threshold
        
    def _update_adaptive_threshold(self):
        if len(self.state.lyapunov_history) > 10:
            recent_v = self.state.lyapunov_history[-10:]
            v_mean = sum(recent_v) / len(recent_v)
            dynamic_offset = min(0.1, v_mean * 0.05)
            self.nerve.alarm_threshold = self.base_threshold + dynamic_offset
            self.nerve.ocgr.alarm_system.threshold = self.nerve.alarm_threshold

    def run(self, on_cycle: Optional[Callable] = None):
        def handle_alarm(alarm_event):
            # CausalNerve calls this callback when an alarm occurs in ocgr.step()
            pass
            
        self.nerve.watch(threshold=self.nerve.alarm_threshold, on_alarm=handle_alarm, auto_revise=True)
        
        for data in self.stream.stream():
            cycle = data["cycle"]
            obs = data["x"]
            
            # 1. Step the nerve
            step_res = self.nerve.step(obs)
            
            # 2. Check for alarms and handle cooldown
            if step_res.alarms_fired > 0:
                if cycle - self._last_alarm_cycle >= self.cooldown_cycles:
                    self._last_alarm_cycle = cycle
                    self.state.log_alarm(cycle, {"alarms_fired": step_res.alarms_fired})
                    
            # 3. Handle asynchronous graph revisions safely
            if step_res.edits_applied > 0:
                # Check history for the newly applied edits
                if self.nerve.ocgr.history.history:
                    for rev in reversed(self.nerve.ocgr.history.history):
                        # We might have processed this before, but in this synchronous setup
                        # we check the cycle matches the current one.
                        if rev.cycle == cycle:
                            self.state.log_surgery(cycle, rev.accepted, rev.to_dict())
                        else:
                            break
            
            # 4. Update state metrics
            health = self.nerve.structural_health()
            self.state.leakage_history.append(health.overall_leakage)
            self.state.lyapunov_history.append(health.v_energy)
            
            self._update_adaptive_threshold()
            
            adj = self.nerve.graph.get_dense_adjacency()
            self.state.current_edges = int((adj > 0.01).sum().item())
            
            # 5. User callback
            if on_cycle:
                on_cycle(cycle, data, self.state)
