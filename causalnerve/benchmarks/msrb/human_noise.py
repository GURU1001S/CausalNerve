import numpy as np
from typing import Iterator, Tuple, Dict, List, Optional

class HumanInterventionNoiseInjector:
    """
    Injects realistic human operator noise into the data stream.
    
    Real industrial systems have humans who:
    - Perform maintenance at unexpected times
    - Apply interventions based on incomplete information
    - Override automated systems incorrectly
    - Document actions inconsistently or not at all
    """
    
    def __init__(self, stream: Iterator[np.ndarray], seed: int = 42):
        self.stream = stream
        self.rng = np.random.default_rng(seed)
        self.active_pathologies = {}
    
    def random_operator_override(self,
                                   cycle: int,
                                   affected_nodes: List[int],
                                   override_value: float,
                                   duration: int = 3
                                   ) -> 'HumanInterventionNoiseInjector':
        """
        An operator clamps specific nodes to a value for duration cycles,
        then releases them. This is NOT a do() intervention by CausalNerve —
        it is an external action that disrupts the causal graph.
        
        Effect: leakage spikes when override starts (structure disrupted)
        and again when it ends (structure restored suddenly).
        
        Critical question: does CausalNerve incorrectly try to revise
        its graph structure to accommodate a temporary human override?
        
        Correct behavior: detect the override as a transient artifact,
        not propose permanent structural edits.
        """
        self.active_pathologies[f'override_{cycle}'] = {
            'type': 'override',
            'start': cycle,
            'end': cycle + duration,
            'nodes': affected_nodes,
            'value': override_value
        }
        return self
    
    def contradictory_intervention(self,
                                     node: int,
                                     value_sequence: List[float],
                                     interval_cycles: int = 10
                                     ) -> 'HumanInterventionNoiseInjector':
        """
        An operator applies conflicting values to the same node
        repeatedly: clamp to 0.8, then 0.2, then 0.8, then 0.2.
        
        This simulates: operator uncertainty, trial-and-error maintenance,
        conflicting instructions from different team members.
        
        Effect: creates oscillating leakage that may trigger OCGR's
        edit-and-revert cycle (oscillation).
        
        Test: does the Lyapunov gate correctly suppress oscillation
        even under contradictory human interventions?
        """
        self.active_pathologies[f'contradictory_{node}'] = {
            'type': 'contradictory',
            'node': node,
            'seq': value_sequence,
            'interval': interval_cycles,
            'start_cycle': None # Will be set on first iteration if None, or can just be 0
        }
        return self
    
    def undocumented_topology_change(self,
                                       add_edge: Optional[Tuple] = None,
                                       remove_edge: Optional[Tuple] = None,
                                       at_cycle: int = 100
                                       ) -> 'HumanInterventionNoiseInjector':
        """
        Simulates: physical maintenance that changes the system topology
        without any notification to the monitoring system.
        
        A sensor is physically rerouted. A component is replaced.
        The causal graph changes in the real world but CausalNerve
        is not told.
        
        This is the PRIMARY scenario OCGR was designed for.
        This is the most important realistic noise test.
        
        Measure:
        - How many cycles until CausalNerve detects the change?
        - Does it correctly identify the changed edge?
        - Is the detection delay <= 20 cycles?
        """
        self.active_pathologies[f'topology_change_{at_cycle}'] = {
            'type': 'topology_change',
            'at_cycle': at_cycle,
            'add': add_edge,
            'remove': remove_edge,
            'history': []
        }
        return self

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, Dict[str, bool]]]:
        cycle = 0
        
        for observation in self.stream:
            corrupted = observation.copy()
            labels = {}
            
            for name, p in self.active_pathologies.items():
                ptype = p['type']
                is_active = False
                
                if ptype == 'override':
                    if p['start'] <= cycle < p['end']:
                        for n in p['nodes']:
                            if n < len(corrupted):
                                corrupted[n] = p['value']
                        is_active = True
                        
                elif ptype == 'contradictory':
                    if p['start_cycle'] is None:
                        p['start_cycle'] = cycle
                    
                    elapsed = cycle - p['start_cycle']
                    seq_idx = (elapsed // p['interval']) % len(p['seq'])
                    node = p['node']
                    if node < len(corrupted):
                        corrupted[node] = p['seq'][seq_idx]
                    is_active = True
                    
                elif ptype == 'topology_change':
                    # To accurately simulate an undocumented topology change on the stream,
                    # we must inject the effect of add_edge or remove the effect of remove_edge.
                    # Since we wrap the final stream, this relies on a simple linear additive assumption.
                    p['history'].append(observation.copy())
                    
                    if cycle >= p['at_cycle']:
                        is_active = True
                        if p['add']:
                            src, tgt = p['add']
                            if src < len(corrupted) and tgt < len(corrupted):
                                # Add 0.5 * source to target
                                corrupted[tgt] += 0.5 * corrupted[src]
                                
                        if p['remove']:
                            src, tgt = p['remove']
                            if src < len(corrupted) and tgt < len(corrupted):
                                # Naively subtract assuming original weight was ~0.5.
                                # (In a true simulation this would be generated natively).
                                corrupted[tgt] -= 0.5 * corrupted[src]
                
                labels[name] = is_active
                
            yield corrupted, cycle, labels
            cycle += 1
