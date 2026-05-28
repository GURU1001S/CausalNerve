import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

class MacroCausalState(Enum):
    NOMINAL = "Nominal Operation"
    COMBUSTION_INSTABILITY = "Combustion Instability"
    COMPRESSOR_DEGRADATION = "Compressor Degradation"
    THERMAL_RUNAWAY = "Thermal Runaway"
    OSCILLATORY_INSTABILITY = "Oscillatory Instability"
    PRESSURE_IMBALANCE = "Pressure Imbalance"
    UNCERTAIN = "Uncertain Motif"

@dataclass
class StructuralEvent:
    name: str
    edges_involved: List[Tuple[int, int]]
    confidence: float
    description: str

class MotifCompressor:
    """Compresses low-level edge probabilities into high-level structural events."""
    
    # Node subsets
    COMPRESSOR_NODES = {0, 1, 2}
    COMBUSTOR_NODES = {3, 11}
    TURBINE_NODES = {4, 5, 9}
    PRESSURE_NODES = {8, 10}

    @classmethod
    def compress(cls, edge_matrix: List[List[float]], threshold: float = 0.5) -> List[StructuralEvent]:
        n = len(edge_matrix)
        events = []
        
        # Identify strong subgraphs
        comp_edges = []
        comb_edges = []
        turb_edges = []
        press_edges = []
        
        for i in range(n):
            for j in range(n):
                if i != j and edge_matrix[i][j] >= threshold:
                    conf = edge_matrix[i][j]
                    if i in cls.COMPRESSOR_NODES and j in cls.COMPRESSOR_NODES:
                        comp_edges.append((i, j, conf))
                    if i in cls.COMBUSTOR_NODES or j in cls.COMBUSTOR_NODES:
                        comb_edges.append((i, j, conf))
                    if i in cls.TURBINE_NODES and j in cls.TURBINE_NODES:
                        turb_edges.append((i, j, conf))
                    if i in cls.PRESSURE_NODES or j in cls.PRESSURE_NODES:
                        press_edges.append((i, j, conf))

        if len(comp_edges) >= 2:
            avg_conf = sum(c for _, _, c in comp_edges) / len(comp_edges)
            events.append(StructuralEvent(
                name="Compressor Stage Cascade",
                edges_involved=[(i, j) for i, j, _ in comp_edges],
                confidence=avg_conf,
                description=f"Cascading causal dependence across {len(comp_edges)} compressor links."
            ))
            
        if len(comb_edges) >= 1:
            avg_conf = sum(c for _, _, c in comb_edges) / len(comb_edges)
            events.append(StructuralEvent(
                name="Combustion Dynamics Shift",
                edges_involved=[(i, j) for i, j, _ in comb_edges],
                confidence=avg_conf,
                description="Altered causal flow originating in or affecting combustion stages."
            ))
            
        if len(turb_edges) >= 2:
            avg_conf = sum(c for _, _, c in turb_edges) / len(turb_edges)
            events.append(StructuralEvent(
                name="Turbine Thermal Coupling",
                edges_involved=[(i, j) for i, j, _ in turb_edges],
                confidence=avg_conf,
                description="High thermal correlation across turbine/cooling subsystems."
            ))

        return events

class TemporalPatternMiner:
    """Detects recurring trajectories and oscillatory instability."""
    def __init__(self, history_size: int = 10):
        self.history: List[List[List[float]]] = []
        self.max_size = history_size

    def update(self, edge_matrix: List[List[float]]):
        self.history.append(edge_matrix)
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def detect_oscillatory_instability(self, threshold: float = 0.3) -> bool:
        if len(self.history) < 3:
            return False
            
        # Check variance of edge probabilities over time
        arr = np.array(self.history)
        variances = np.var(arr, axis=0)
        
        # If any edge has high variance (constantly flipping)
        if np.max(variances) > threshold * threshold:
            return True
        return False

class HierarchicalGraphSummarizer:
    """Translates graph metrics and events into a dominant state."""
    
    @staticmethod
    def identify_macro_state(events: List[StructuralEvent], is_oscillating: bool) -> MacroCausalState:
        if is_oscillating:
            return MacroCausalState.OSCILLATORY_INSTABILITY
            
        if not events:
            return MacroCausalState.NOMINAL
            
        # Sort events by confidence to find dominant
        events.sort(key=lambda e: e.confidence, reverse=True)
        dominant = events[0].name
        
        if "Combustion" in dominant:
            return MacroCausalState.COMBUSTION_INSTABILITY
        elif "Compressor" in dominant:
            return MacroCausalState.COMPRESSOR_DEGRADATION
        elif "Turbine" in dominant:
            return MacroCausalState.THERMAL_RUNAWAY
        elif "Pressure" in dominant:
            return MacroCausalState.PRESSURE_IMBALANCE
            
        return MacroCausalState.UNCERTAIN

class NarrativeEngine:
    """Generates symbolic, deterministic textual reasoning."""
    
    @staticmethod
    def generate_narrative(state: MacroCausalState, events: List[StructuralEvent], n_edges: int) -> str:
        if state == MacroCausalState.NOMINAL:
            return f"Engine operating normally. Graph density stable at {n_edges} edges."
            
        if state == MacroCausalState.OSCILLATORY_INSTABILITY:
            return "Critical alert: Causal structure is rapidly oscillating, indicating severe non-stationary degradation."
            
        if not events:
            return f"Graph topology shifted ({n_edges} edges), resulting in {state.value}."
            
        dominant_event = max(events, key=lambda e: e.confidence)
        
        text = f"Instead of tracking {n_edges} low-level edges, abstraction engine detected {state.value}. "
        text += f"Dominant motif is '{dominant_event.name}' (Confidence: {dominant_event.confidence:.2f}). "
        text += dominant_event.description
        
        return text

class AbstractionLayer:
    """Orchestrates the entire causal abstraction process."""
    def __init__(self):
        self.miner = TemporalPatternMiner()

    def process(self, edge_matrix: List[List[float]], threshold: float = 0.5) -> Dict[str, Any]:
        self.miner.update(edge_matrix)
        
        events = MotifCompressor.compress(edge_matrix, threshold)
        is_oscillating = self.miner.detect_oscillatory_instability()
        
        macro_state = HierarchicalGraphSummarizer.identify_macro_state(events, is_oscillating)
        
        # Count active edges
        n_edges = sum(1 for i in range(len(edge_matrix)) for j in range(len(edge_matrix)) if i != j and edge_matrix[i][j] >= threshold)
        
        narrative = NarrativeEngine.generate_narrative(macro_state, events, n_edges)
        
        dominant_motif = events[0].name if events else "None"
        
        return {
            "macro_state": macro_state.value,
            "narrative": narrative,
            "dominant_motif": dominant_motif,
            "events": [e.name for e in events],
            "is_oscillating": is_oscillating
        }
