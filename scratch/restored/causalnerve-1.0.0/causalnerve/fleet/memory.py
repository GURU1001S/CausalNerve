"""
causalnerve.fleet.memory
========================
Converts the revision database into predictive structural memory.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .database import FleetRevisionDatabase
from ..adapt.ocgr import RevisionEvent

@dataclass
class AssetFingerprint:
    asset_id: str
    revision_sequence: List[Tuple[Tuple[int, int], str]]
    state_trajectory: List[List[float]]
    confirmed_hypotheses: List[str]

@dataclass
class SimilarAsset:
    asset_id: str
    similarity_score: float
    fingerprint: AssetFingerprint

@dataclass
class PredictedTransition:
    edge: Tuple[int, int]
    edit_type: str
    predicted_cycles_from_now: int
    confidence: float
    supporting_asset_ids: List[str]
    state_condition_at_event: List[float]

class StructuralRecurrenceMemory:
    """
    Converts the revision database into predictive structural memory.
    """
    
    def build_fingerprint(self,
                          asset_id: str,
                          db: FleetRevisionDatabase) -> AssetFingerprint:
        """
        Build structural biography of an asset.
        """
        query = "SELECT src_node, dst_node, edit_type, state_snapshot FROM revisions WHERE asset_id = ? ORDER BY cycle ASC"
        cursor = db.conn.execute(query, (asset_id,))
        
        sequence = []
        trajectory = []
        for row in cursor.fetchall():
            sequence.append(((row[0], row[1]), row[2]))
            import json
            state = json.loads(row[3]) if row[3] != "{}" else []
            trajectory.append(state)
            
        return AssetFingerprint(
            asset_id=asset_id,
            revision_sequence=sequence,
            state_trajectory=trajectory,
            confirmed_hypotheses=[]
        )

    def _dtw_distance(self, seq1: List[List[float]], seq2: List[List[float]]) -> float:
        """Simple DTW implementation for variable length trajectories."""
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return float('inf')
            
        dtw_matrix = np.full((n+1, m+1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(np.array(seq1[i-1]) - np.array(seq2[j-1]))
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                              dtw_matrix[i, j-1],    # deletion
                                              dtw_matrix[i-1, j-1])  # match
        return float(dtw_matrix[n, m])

    def find_similar_assets(self,
                            fingerprint: AssetFingerprint,
                            db: FleetRevisionDatabase,
                            top_k: int = 5,
                            similarity: str = "dtw") -> List[SimilarAsset]:
        """
        Find K most similar assets by trajectory fingerprint.
        """
        # Get all distinct assets
        cursor = db.conn.execute("SELECT DISTINCT asset_id FROM revisions WHERE asset_id != ?", (fingerprint.asset_id,))
        other_assets = [row[0] for row in cursor.fetchall()]
        
        if not other_assets:
            return []
            
        similarities = []
        for other_id in other_assets:
            other_fp = self.build_fingerprint(other_id, db)
            
            if similarity == "dtw":
                score = 1.0 / (1.0 + self._dtw_distance(fingerprint.state_trajectory, other_fp.state_trajectory))
            elif similarity == "edit":
                # Jaccard on edit sets for simplicity in this architecture
                set1 = set(fingerprint.revision_sequence)
                set2 = set(other_fp.revision_sequence)
                if not set1 and not set2:
                    score = 1.0
                else:
                    score = len(set1.intersection(set2)) / len(set1.union(set2))
            else:
                score = 0.0
                
            similarities.append(SimilarAsset(other_id, score, other_fp))
            
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]

    def predict_next_transition(self,
                                fingerprint: AssetFingerprint,
                                similar_assets: List[SimilarAsset]) -> List[PredictedTransition]:
        """
        For K similar assets: what structural transitions did 
        they experience next after reaching a similar state?
        """
        if not similar_assets:
            return []
            
        current_len = len(fingerprint.revision_sequence)
        
        predictions = []
        # Look at the transition that happened in similar assets at current_len + 1
        for asset in similar_assets:
            seq = asset.fingerprint.revision_sequence
            if len(seq) > current_len:
                next_edit = seq[current_len]
                edge, edit_type = next_edit
                
                # In a real impl, cycles from now is computed from db cycle timestamps
                predictions.append(
                    PredictedTransition(
                        edge=edge,
                        edit_type=edit_type,
                        predicted_cycles_from_now=50, 
                        confidence=asset.similarity_score,
                        supporting_asset_ids=[asset.asset_id],
                        state_condition_at_event=asset.fingerprint.state_trajectory[current_len] if len(asset.fingerprint.state_trajectory) > current_len else []
                    )
                )
                
        return predictions
