import numpy as np
from typing import Iterator, Optional, List, Dict, Any, Tuple

class SensorPathologyInjector:
    """
    Wraps any data stream and injects realistic sensor corruption.
    Domain-agnostic: works on any (T, N_features) stream.
    
    Usage:
        clean_stream = SyntheticStreamGenerator.with_drift(drift_at=100)
        messy_stream = SensorPathologyInjector(clean_stream).inject_all()
        nerve.watch(messy_stream)
    """
    
    def __init__(self, stream: Iterator[np.ndarray], seed: int = 42):
        self.stream = stream
        self.rng = np.random.default_rng(seed)
        self.active_pathologies = {}
    
    def stuck_sensor(self,
                      sensor_idx: int,
                      start_cycle: int,
                      duration: int,
                      stuck_value: Optional[float] = None
                      ) -> 'SensorPathologyInjector':
        """
        Sensor reports the same value for `duration` cycles.
        stuck_value: if None, uses the value at start_cycle.
        
        This is extremely common in industrial deployments:
        a sensor freezes, usually at its last reading or at zero.
        
        Effect on CausalNerve: should trigger dropout_artifact_detection
        if the stuck sensor causes phantom leakage spikes.
        """
        self.active_pathologies[f'stuck_{sensor_idx}'] = {
            'type': 'stuck',
            'sensor': sensor_idx,
            'start': start_cycle,
            'end': start_cycle + duration,
            'stuck_value': stuck_value
        }
        return self
    
    def drifting_calibration(self,
                              sensor_idx: int,
                              drift_rate: float = 0.001,
                              drift_direction: str = 'positive'
                              ) -> 'SensorPathologyInjector':
        """
        Sensor reading drifts slowly from true value.
        drift_rate: per-cycle additive drift (0.001 = 0.1% per 100 cycles)
        
        This is the hardest pathology to detect because it looks
        like a genuine slow structural change. The drift discriminator
        must distinguish calibration drift from genuine coupling emergence.
        
        Ground truth: any edit proposed during calibration drift
        that would not be proposed on clean data = false surgery.
        """
        self.active_pathologies[f'drift_{sensor_idx}'] = {
            'type': 'calibration_drift',
            'sensor': sensor_idx,
            'rate': drift_rate if drift_direction == 'positive' else -drift_rate,
            'cumulative': 0.0
        }
        return self
    
    def quantization_artifact(self,
                               sensor_idx: int,
                               bits: int = 4
                               ) -> 'SensorPathologyInjector':
        """
        Reduce sensor to `bits`-bit quantization.
        Creates artificial discretization steps that look like
        structural changes to leakage-based detectors.
        
        4-bit: very coarse (common in embedded IoT sensors)
        8-bit: moderate (common in SCADA systems)
        """
        self.active_pathologies[f'quant_{sensor_idx}'] = {
            'type': 'quantization',
            'sensor': sensor_idx,
            'levels': 2**bits
        }
        return self
    
    def intermittent_silence(self,
                              sensor_idx: int,
                              silence_probability: float = 0.05
                              ) -> 'SensorPathologyInjector':
        """
        Each cycle: sensor reports NaN with probability silence_probability.
        Replaced with last valid reading (zero-order hold).
        
        5% silence = 1 in 20 cycles missing (realistic IoT)
        20% silence = 1 in 5 cycles missing (degraded network)
        """
        self.active_pathologies[f'silence_{sensor_idx}'] = {
            'type': 'intermittent_silence',
            'sensor': sensor_idx,
            'prob': silence_probability,
            'last_valid': None
        }
        return self
    
    def burst_corruption(self,
                          start_cycle: int,
                          duration: int = 5,
                          affected_sensors: Optional[List[int]] = None,
                          corruption_magnitude: float = 3.0
                          ) -> 'SensorPathologyInjector':
        """
        All specified sensors simultaneously corrupted for `duration` cycles.
        Values replaced with random N(0, corruption_magnitude) noise.
        
        Simulates: packet collision, electrical interference, power spike.
        
        Critical test: does CausalNerve's simultaneity rule in
        DropoutArtifactDetector correctly classify this as an artifact?
        If it proposes real edits during a burst: false surgery.
        """
        sensors = affected_sensors or list(range(6))
        self.active_pathologies[f'burst_{start_cycle}'] = {
            'type': 'burst',
            'start': start_cycle,
            'end': start_cycle + duration,
            'sensors': sensors,
            'magnitude': corruption_magnitude
        }
        return self
    
    def duplicated_channel(self,
                            source_idx: int,
                            target_idx: int,
                            noise_scale: float = 0.02
                            ) -> 'SensorPathologyInjector':
        """
        target_idx sensor reports source_idx values plus small noise.
        Simulates: misconfigured data pipeline, channel remapping error.
        
        Effect: creates spurious apparent correlation between two nodes.
        The causal sufficiency checker must detect this:
        source and target are not causally related — they share a common input.
        """
        self.active_pathologies[f'dup_{source_idx}_{target_idx}'] = {
            'type': 'duplicated_channel',
            'source': source_idx,
            'target': target_idx,
            'noise': noise_scale
        }
        return self
    
    def inject_all(self,
                    preset: str = 'moderate'
                    ) -> Iterator[Tuple[np.ndarray, int, Dict[str, bool]]]:
        """
        Apply a preset combination of pathologies.
        
        preset options:
            'mild':     stuck sensor (1), calibration drift (1), 5% silence
            'moderate': all of the above + burst (2), quantization (2 sensors)
            'severe':   all pathologies at high intensity
            'adversarial': pathologies designed to maximally confuse 
                          causal leakage detection
        
        The 'adversarial' preset is most important:
        it tests whether CausalNerve proposes false edits under
        the worst realistic corruption pattern.
        """
        presets = {
            'mild': self._apply_mild,
            'moderate': self._apply_moderate,
            'severe': self._apply_severe,
            'adversarial': self._apply_adversarial
        }
        presets[preset]()
        return iter(self)

    def _apply_mild(self):
        self.stuck_sensor(1, 50, 20)
        self.drifting_calibration(0, 0.001)
        self.intermittent_silence(2, 0.05)

    def _apply_moderate(self):
        self._apply_mild()
        self.burst_corruption(120, 5, [0, 1])
        self.quantization_artifact(3, 8)
        self.quantization_artifact(4, 8)

    def _apply_severe(self):
        self.stuck_sensor(1, 50, 100)
        self.drifting_calibration(0, 0.01)
        self.intermittent_silence(2, 0.20)
        self.burst_corruption(80, 10, None, 5.0)
        self.quantization_artifact(3, 4)
        self.duplicated_channel(0, 5, 0.01)
    
    def _apply_adversarial(self):
        """
        The hardest realistic test:
        - Calibration drift on the PRIMARY leakage-monitored sensor
          (makes drift look like genuine structural change)
        - Duplicated channel on two causally unrelated nodes
          (creates spurious correlations)
        - Burst corruption timed to coincide with a real structural event
          (confounds genuine alarm with noise artifact)
        - Stuck sensor at a value near the alarm threshold
          (keeps leakage permanently near but not at threshold)
        """
        self.drifting_calibration(0, drift_rate=0.005)
        self.duplicated_channel(2, 4, noise_scale=0.01)
        self.burst_corruption(100, duration=3, affected_sensors=[0, 1, 2], corruption_magnitude=4.0)
        self.stuck_sensor(5, start_cycle=60, duration=200, stuck_value=0.85)

    def _apply_pathology(self, obs: np.ndarray, p: Dict[str, Any], cycle: int) -> np.ndarray:
        ptype = p['type']
        
        if ptype == 'stuck':
            if p['start'] <= cycle < p['end']:
                if p['stuck_value'] is None:
                    p['stuck_value'] = obs[p['sensor']]
                obs[p['sensor']] = p['stuck_value']
                
        elif ptype == 'calibration_drift':
            p['cumulative'] += p['rate']
            obs[p['sensor']] += p['cumulative']
            
        elif ptype == 'quantization':
            levels = p['levels']
            obs[p['sensor']] = np.round(obs[p['sensor']] * levels) / levels
            
        elif ptype == 'intermittent_silence':
            if self.rng.random() < p['prob']:
                if p['last_valid'] is not None:
                    obs[p['sensor']] = p['last_valid']
                else:
                    obs[p['sensor']] = 0.0 # fallback if first observation is NaN
            else:
                p['last_valid'] = obs[p['sensor']]
                
        elif ptype == 'burst':
            if p['start'] <= cycle < p['end']:
                for s in p['sensors']:
                    if s < len(obs):
                        obs[s] = self.rng.normal(0, p['magnitude'])
                        
        elif ptype == 'duplicated_channel':
            src = p['source']
            tgt = p['target']
            if src < len(obs) and tgt < len(obs):
                obs[tgt] = obs[src] + self.rng.normal(0, p['noise'])
                
        return obs
        
    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, Dict[str, bool]]]:
        """Wrap the underlying stream with active pathologies."""
        cycle = 0
        for observation in self.stream:
            corrupted = observation.copy()
            for name, p in self.active_pathologies.items():
                corrupted = self._apply_pathology(corrupted, p, cycle)
            
            labels = self._active_pathology_labels(cycle)
            cycle += 1
            yield corrupted, cycle, labels
    
    def _active_pathology_labels(self, cycle: int) -> Dict[str, bool]:
        """
        Returns which pathologies are active this cycle.
        Used by evaluator to compute false surgery rate correctly:
        an edit during an active pathology = potential false surgery.
        """
        active = {}
        for name, p in self.active_pathologies.items():
            ptype = p['type']
            is_active = False
            
            if ptype in ('stuck', 'burst'):
                is_active = (p['start'] <= cycle < p['end'])
            elif ptype == 'intermittent_silence':
                # The label here could indicate if it *was* silenced this cycle
                # But typically we just say the pathology is active globally
                is_active = True 
            elif ptype in ('calibration_drift', 'quantization', 'duplicated_channel'):
                is_active = True
                
            active[name] = is_active
            
        return active
