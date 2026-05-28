import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from .live_memory import FleetStructuralMemory

def compute_dtw(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Lightweight Dynamic Time Warping (DTW) for comparing 
    thermodynamic signatures (leakage, energy, uncertainty).
    Uses Euclidean distance for 1D/multi-dimensional points.
    """
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return float('inf')
        
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # deletion
                dtw_matrix[i - 1, j - 1] # match
            )
            
    return float(dtw_matrix[n, m])

class StructuralPrecognitionEngine:
    """
    Predicts likely future graph edits by comparing live trajectory 
    signatures against the FleetStructuralMemory using DTW.
    """
    def __init__(self, memory: FleetStructuralMemory, signature_window: int = 15):
        self.memory = memory
        self.signature_window = signature_window
        
    def _extract_signature(self, leakage: List[float], energy: List[float], unc: List[float]) -> np.ndarray:
        """Fuses metrics into a unified 3D thermodynamic signature array."""
        l_arr = np.array(leakage[-self.signature_window:])
        e_arr = np.array(energy[-self.signature_window:])
        u_arr = np.array(unc[-self.signature_window:])
        
        # Pad if too short
        if len(l_arr) < self.signature_window:
            pad = self.signature_window - len(l_arr)
            l_arr = np.pad(l_arr, (pad, 0), mode='edge')
            e_arr = np.pad(e_arr, (pad, 0), mode='edge')
            u_arr = np.pad(u_arr, (pad, 0), mode='edge')
            
        # Z-score normalize locally to capture shape/dynamics rather than absolute magnitude
        l_arr = (l_arr - np.mean(l_arr)) / (np.std(l_arr) + 1e-6)
        e_arr = (e_arr - np.mean(e_arr)) / (np.std(e_arr) + 1e-6)
        u_arr = (u_arr - np.mean(u_arr)) / (np.std(u_arr) + 1e-6)
        
        return np.column_stack((l_arr, e_arr, u_arr))
        
    def predict_next_surgery(self, current_leakage: List[float], current_energy: List[float], current_unc: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Scans fleet memory for similar structural precursor states and outputs predictions.
        """
        if len(current_leakage) < 5:
            return []
            
        current_sig = self._extract_signature(current_leakage, current_energy, current_unc)
        past_events = self.memory.get_all_accepted_surgeries()
        
        if not past_events:
            return []
            
        scored_events = []
        for event in past_events:
            hist_sig = self._extract_signature(event['leakage_hist'], event['energy_hist'], event['uncertainty_hist'])
            distance = compute_dtw(current_sig, hist_sig)
            
            # Convert DTW distance to a similarity score (0 to 1)
            similarity = np.exp(-distance / (self.signature_window * 1.5))
            
            scored_events.append({
                'similarity': similarity,
                'distance': distance,
                'event': event
            })
            
        # Sort by similarity descending
        scored_events.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Aggregate probabilities by surgery action
        surgery_probs = defaultdict(float)
        surgery_engines = defaultdict(set)
        
        total_weight = 0.0
        for se in scored_events[:10]: # Look at top 10 matches to form probability distribution
            sim = se['similarity']
            if sim < 0.1: continue
            
            action_key = (se['event']['edit_type'], se['event']['edge'])
            surgery_probs[action_key] += sim
            surgery_engines[action_key].add(se['event']['engine_id'])
            total_weight += sim
            
        predictions = []
        for action, weight in surgery_probs.items():
            prob = weight / total_weight if total_weight > 0 else 0
            if prob > 0:
                predictions.append({
                    'edit_type': action[0],
                    'edge': action[1],
                    'probability': prob,
                    'precursor_similarity': weight / max(1, len(surgery_engines[action])), # avg similarity
                    'matched_historical_engines': list(surgery_engines[action])
                })
                
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions[:top_k]
