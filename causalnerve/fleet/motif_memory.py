import json
import os
import threading
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

class MotifFingerprint:
    """
    Converts local subgraphs into canonical, hashable fingerprints.
    Tolerates minor edge weight variations and drops weak edges.
    """
    @staticmethod
    def generate(edge_matrix: List[List[float]], threshold: float = 0.1) -> str:
        """
        Generate a structural fingerprint from an edge probability matrix.
        Only considers edges with probability >= threshold.
        """
        n = len(edge_matrix)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and edge_matrix[i][j] >= threshold:
                    edges.append((i, j))
        
        # Sort edges to ensure canonical ordering
        edges.sort()
        
        # Create a string representation and hash it
        edge_str = "|".join([f"{u}-{v}" for u, v in edges])
        return hashlib.sha256(edge_str.encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def jaccard_similarity(mat_a: List[List[float]], mat_b: List[List[float]], threshold: float = 0.1) -> float:
        """Computes Jaccard similarity between two edge matrices based on active edges."""
        n = len(mat_a)
        set_a = {(i, j) for i in range(n) for j in range(n) if i != j and mat_a[i][j] >= threshold}
        set_b = {(i, j) for i in range(n) for j in range(n) if i != j and mat_b[i][j] >= threshold}
        
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union


class MotifMemoryBank:
    """
    Thread-safe persistent storage for discovered graph motifs across the fleet.
    """
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            home = str(Path.home())
            self.storage_path = os.path.join(home, '.causalnerve', 'motif_memory.json')
        else:
            self.storage_path = storage_path
            
        self.lock = threading.RLock()
        self.motifs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
                if os.path.exists(self.storage_path):
                    with open(self.storage_path, 'r', encoding='utf-8') as f:
                        self.motifs = json.load(f)
            except Exception as e:
                print(f"[MotifMemoryBank] Error loading memory: {e}")
                self.motifs = {}

    def _save(self):
        with self.lock:
            try:
                with open(self.storage_path, 'w', encoding='utf-8') as f:
                    json.dump(self.motifs, f, indent=2)
            except Exception as e:
                print(f"[MotifMemoryBank] Error saving memory: {e}")

    def add_or_update_motif(self, edge_matrix: List[List[float]], 
                            engine_id: str,
                            leakage_signature: float,
                            lyapunov_trajectory: float,
                            cycles_to_failure_improvement: int,
                            supporting_sensors: List[float],
                            intervention_success: bool) -> str:
        with self.lock:
            fingerprint = MotifFingerprint.generate(edge_matrix)
            
            if fingerprint in self.motifs:
                m = self.motifs[fingerprint]
                m['confirmations'] += 1
                if engine_id not in m['engines_observed']:
                    m['engines_observed'].append(engine_id)
                
                # Exponential moving average for metrics
                alpha = 0.2
                m['avg_leakage'] = (1 - alpha) * m['avg_leakage'] + alpha * leakage_signature
                m['avg_lyapunov'] = (1 - alpha) * m['avg_lyapunov'] + alpha * lyapunov_trajectory
                m['mean_cycles_improvement'] = (1 - alpha) * m['mean_cycles_improvement'] + alpha * cycles_to_failure_improvement
                
                if intervention_success:
                    m['success_count'] += 1
                
                m['intervention_success_rate'] = m['success_count'] / m['confirmations']
                m['confidence_score'] = min(1.0, m['confirmations'] / 10.0) * m['intervention_success_rate']
            else:
                self.motifs[fingerprint] = {
                    'fingerprint': fingerprint,
                    'edge_matrix': edge_matrix,
                    'supporting_sensors': supporting_sensors,
                    'avg_leakage': leakage_signature,
                    'avg_lyapunov': lyapunov_trajectory,
                    'mean_cycles_improvement': cycles_to_failure_improvement,
                    'confirmations': 1,
                    'success_count': 1 if intervention_success else 0,
                    'intervention_success_rate': 1.0 if intervention_success else 0.0,
                    'confidence_score': 0.1 * (1.0 if intervention_success else 0.0),
                    'engines_observed': [engine_id]
                }
                
            self._save()
            return fingerprint

    def get_all(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.motifs.values())

    def clear(self):
        with self.lock:
            self.motifs = {}
            self._save()


class MotifMatcher:
    """
    Searches the memory bank for subgraphs matching the live engine state.
    """
    def __init__(self, memory_bank: MotifMemoryBank):
        self.memory_bank = memory_bank

    def find_matches(self, current_edge_matrix: List[List[float]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        matches = []
        for motif in self.memory_bank.get_all():
            sim = MotifFingerprint.jaccard_similarity(current_edge_matrix, motif['edge_matrix'])
            if sim >= threshold:
                match_data = motif.copy()
                match_data['similarity'] = sim
                matches.append(match_data)
        
        matches.sort(key=lambda x: x['similarity'] * x['confidence_score'], reverse=True)
        return matches


class EarlyWarningEngine:
    """
    Monitors live data against historical motifs to preempt structural failures.
    """
    def __init__(self, matcher: MotifMatcher):
        self.matcher = matcher

    def evaluate(self, current_edge_matrix: List[List[float]], current_leakage: float) -> Optional[Dict[str, Any]]:
        matches = self.matcher.find_matches(current_edge_matrix, threshold=0.6)
        if not matches:
            return None
            
        best_match = matches[0]
        # If the motif represents a high-confidence recurring failure state
        if best_match['confidence_score'] > 0.6 and current_leakage > (best_match['avg_leakage'] * 0.5):
            return {
                "warning_triggered": True,
                "motif_fingerprint": best_match['fingerprint'],
                "similarity": best_match['similarity'],
                "transfer_confidence": best_match['confidence_score'],
                "expected_cycles_gained": best_match['mean_cycles_improvement'],
                "previously_seen_in": best_match['engines_observed'],
                "message": f"Precognitive Motif Match: Structural failure pattern detected early (Similarity: {best_match['similarity']:.2f}). Triggered by {len(best_match['engines_observed'])} prior engine trajectories."
            }
            
        return None
