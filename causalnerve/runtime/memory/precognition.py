"""
causalnerve.runtime.memory.precognition
==============================
Early warning signal detection for causal phase transitions.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from ..adaptation.ocgr import RevisionEvent

@dataclass
class PrecursorSignature:
    slope_50: float
    slope_100: float
    acceleration: float
    state_levels: np.ndarray

@dataclass
class DetectionWindowStats:
    mean_cycles_before: float
    std_cycles_before: float

@dataclass
class PrecursorMatch:
    confidence: float
    predicted_cycles_to_event: int

class PrecursorSignatureLibrary:
    """
    For each confirmed revision event type, stores the 
    characteristic state trajectory shape in the N cycles
    before the event. The precursor signature.
    """
    def __init__(self):
        self.signatures: Dict[Tuple[int, int], PrecursorSignature] = {}
        
    def extract_signature(self,
                          event: RevisionEvent,
                          state_history: torch.Tensor,
                          lookback: int = 100) -> PrecursorSignature:
        """
        Extract state trajectory features in lookback window before the event.
        """
        hist = state_history.detach().cpu().numpy()
        if len(hist) < lookback:
            # Fallback if history is too short
            window = hist
        else:
            window = hist[-lookback:]
            
        # Simplified feature extraction for architecture definition
        if len(window) > 50:
            slope_50 = float(np.mean(np.gradient(window[-50:], axis=0)))
        else:
            slope_50 = 0.0
            
        slope_100 = float(np.mean(np.gradient(window, axis=0))) if len(window) > 1 else 0.0
        accel = float(np.mean(np.gradient(np.gradient(window, axis=0), axis=0))) if len(window) > 2 else 0.0
        levels = window[-1] if len(window) > 0 else np.array([])
        
        sig = PrecursorSignature(slope_50, slope_100, accel, levels)
        self.signatures[event.edge] = sig
        return sig

    def detection_window(self, edge: Tuple[int, int]) -> DetectionWindowStats:
        """
        For a given edge type: how many cycles before the event
        is the precursor detectable?
        """
        # In a fully fleshed out database, we query historical detectability.
        # Mock values for architecture.
        return DetectionWindowStats(mean_cycles_before=35.0, std_cycles_before=5.2)

    def match(self,
              current_trajectory: torch.Tensor,
              edge: Tuple[int, int],
              threshold: float = 0.7) -> Optional[PrecursorMatch]:
        """
        Does the current trajectory match the precursor signature
        for this edge type?
        """
        if edge not in self.signatures:
            return None
            
        target_sig = self.signatures[edge]
        hist = current_trajectory.detach().cpu().numpy()
        
        # Extract features of current trajectory
        current_slope = float(np.mean(np.gradient(hist[-50:], axis=0))) if len(hist) > 50 else 0.0
        
        # Simple similarity check
        diff = abs(current_slope - target_sig.slope_50)
        confidence = max(0.0, 1.0 - diff)
        
        if confidence > threshold:
            stats = self.detection_window(edge)
            return PrecursorMatch(
                confidence=confidence,
                predicted_cycles_to_event=int(stats.mean_cycles_before)
            )
            
        return None
