import numpy as np
from typing import Iterator, Tuple, Dict, List, Optional
import collections

class TemporalDisorderInjector:
    """
    Injects temporal pathologies into any streaming data.
    Real telemetry is rarely perfectly synchronized.
    """
    
    def __init__(self, stream: Iterator, seed: int = 42):
        self.stream = stream
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.active_pathologies = {}
    
    def packet_reorder(self,
                        max_delay_cycles: int = 5
                        ) -> 'TemporalDisorderInjector':
        """
        Each observation may be delayed by 0-max_delay_cycles cycles,
        causing out-of-order delivery.
        
        Implementation: maintain a buffer of max_delay_cycles observations.
        On each step: randomly release observations from the buffer.
        Some cycles: deliver 0 observations (gap). Others: deliver 2+.
        
        Effect on CausalNerve:
        Temporal ordering assumption is violated.
        The precursor_absence_check in DropoutArtifactDetector
        may misclassify reordered genuine signals as artifacts.
        Measure: how many genuine alarms are suppressed by reordering?
        """
        self.active_pathologies['packet_reorder'] = {
            'type': 'packet_reorder',
            'max_delay': max_delay_cycles,
            'hold_buffer': collections.deque()
        }
        return self
    
    def timestamp_skew(self,
                        skew_std_cycles: float = 2.0
                        ) -> 'TemporalDisorderInjector':
        """
        Each observation's effective timestamp is shifted by N(0, skew_std)
        where skew_std is in units of cycles.
        
        Simulates: unsynchronized clocks across sensors,
        NTP drift, buffered transmission.
        
        Effect: the causal precursor window (48.6 cycles for HPT->HPC)
        becomes blurred. Measure: does precognition advantage decrease?
        """
        self.active_pathologies['timestamp_skew'] = {
            'type': 'timestamp_skew',
            'skew_std': skew_std_cycles
        }
        return self
    
    def variable_sampling_rate(self,
                                 fast_sensors: List[int],
                                 slow_sensors: List[int],
                                 fast_rate: int = 2,
                                 slow_rate: int = 5
                                 ) -> 'TemporalDisorderInjector':
        """
        Different sensors sample at different rates.
        fast_sensors: report every fast_rate cycles
        slow_sensors: report every slow_rate cycles
        
        Between slow-sensor samples: interpolate linearly.
        
        Simulates: mixed-frequency sensor arrays,
        common in real industrial SCADA systems.
        
        Effect: node state updates are no longer synchronized.
        The causal propagation equation assumes synchrony.
        Measure: does async sampling cause spurious leakage?
        """
        self.active_pathologies['variable_sampling'] = {
            'type': 'variable_sampling',
            'fast_sensors': fast_sensors,
            'slow_sensors': slow_sensors,
            'fast_rate': fast_rate,
            'slow_rate': slow_rate,
            'last_slow_vals': {},
            'next_slow_vals': {},
            'last_fast_vals': {}
        }
        return self
    
    def delayed_propagation(self,
                              edge: Tuple[int,int],
                              delay_cycles: int = 8
                              ) -> 'TemporalDisorderInjector':
        """
        The TRUE causal effect of edge (i->j) appears after delay_cycles.
        This violates the Markov assumption of CSC propagation.
        
        This is a KNOWN LIMITATION of CausalNerve (documented in FAILURES.md).
        This benchmark PROVES the limitation is real and measures its magnitude.
        
        Expected outcome:
        CausalNerve will either:
        A. Propose a spurious intermediate node (wrong)
        B. Miss the delayed edge entirely (wrong but less bad)
        C. Detect the edge but with delay_cycles latency (acceptable)
        
        Document which outcome occurs and under what delay magnitudes.
        """
        self.active_pathologies[f'delayed_prop_{edge[0]}_{edge[1]}'] = {
            'type': 'delayed_propagation',
            'edge': edge,
            'delay': delay_cycles,
            'history': collections.deque(maxlen=delay_cycles + 1)
        }
        return self

    def __iter__(self):
        cycle = 0
        
        # Lookahead buffer for interpolation or delay features if needed
        # (For simpler implementation, we can just track history)
        
        # For packet reorder, we might yield multiple items or none.
        # However, CausalNerve expects a simple generator yielding one obs per cycle in watch().
        # To simulate out-of-order, we can just scramble the actual rows being yielded.
        # We will hold some rows and release them randomly, filling gaps with NaNs or holding.
        # But a standard Iterator[np.ndarray] usually yields one array per step.
        # If we yield 0 items, we might yield a None or NaN array. Let's yield a NaN array to represent gap.
        
        for observation in self.stream:
            corrupted = observation.copy()
            labels = {}
            
            for name, p in self.active_pathologies.items():
                labels[name] = True
                ptype = p['type']
                
                if ptype == 'timestamp_skew':
                    # We can't actually change the loop time easily, but we can simulate the 
                    # blurring effect by blending adjacent cycles or adding noise correlated to derivative.
                    # Or we just add random noise proportional to the signal's rate of change.
                    # A true timestamp skew would require returning (timestamp, observation).
                    # Since our stream yields just observation, we'll simulate by adding noise.
                    skew = self.rng.normal(0, p['skew_std'])
                    # Simple proxy: add noise to represent sampling at a skewed time
                    corrupted += self.rng.normal(0, 0.05, size=corrupted.shape) * skew
                    
                elif ptype == 'variable_sampling':
                    # Slow sensors only update every slow_rate cycles.
                    # Fast sensors only update every fast_rate cycles.
                    
                    for sensor in p['fast_sensors']:
                        if cycle % p['fast_rate'] != 0:
                            if sensor in p['last_fast_vals']:
                                corrupted[sensor] = p['last_fast_vals'][sensor]
                        else:
                            p['last_fast_vals'][sensor] = corrupted[sensor]
                            
                    for sensor in p['slow_sensors']:
                        # ZOH for slow sensors (simulating interpolation requires future knowledge, 
                        # so zero-order hold is causal and standard).
                        if cycle % p['slow_rate'] != 0:
                            if sensor in p['last_slow_vals']:
                                corrupted[sensor] = p['last_slow_vals'][sensor]
                        else:
                            p['last_slow_vals'][sensor] = corrupted[sensor]

                elif ptype == 'delayed_propagation':
                    # Add current observation to history
                    p['history'].append(observation.copy())
                    if len(p['history']) > p['delay']:
                        src, tgt = p['edge']
                        old_obs = p['history'][0]
                        # Inject delayed effect: replace current tgt value 
                        # or add the delayed source value's effect. 
                        # Assuming a simple additive linear effect for the benchmark.
                        corrupted[tgt] += 0.5 * old_obs[src]
            
            # Handle packet reorder (simulating jitter/gaps)
            if 'packet_reorder' in self.active_pathologies:
                p = self.active_pathologies['packet_reorder']
                p['hold_buffer'].append(corrupted.copy())
                # Randomly decide to yield a delayed packet
                if len(p['hold_buffer']) > p['max_delay'] or self.rng.random() > 0.5:
                    if p['hold_buffer']:
                        # Shuffle buffer to simulate reordering
                        idx = self.rng.integers(0, len(p['hold_buffer']))
                        corrupted = p['hold_buffer'][idx]
                        del p['hold_buffer'][idx]
                    else:
                        corrupted = np.full_like(corrupted, np.nan)
                else:
                    # Gap
                    corrupted = np.full_like(corrupted, np.nan)
            
            yield corrupted, cycle, labels
            cycle += 1
