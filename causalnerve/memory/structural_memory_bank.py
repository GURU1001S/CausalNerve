from .episodic_memory import EpisodicMemory
from .motif_archive import MotifArchive
from .recurrence_engine import RecurrenceEngine

import numpy as np

class StructuralMemoryBank:
    def __init__(self):
        self.episodic_memory = EpisodicMemory()
        self.motif_archive = MotifArchive()
        self.engine = RecurrenceEngine()
        self._latest_adj = None
        self._metadata = []
        
    def store_regime(self, adj, state=None, states=None, metadata=None):
        _state = state if state is not None else states
        compressed = self.motif_archive.compress(adj)
        self.episodic_memory.add_episode(compressed, _state)
        self._latest_adj = compressed
        self._metadata.append(metadata)
        
    def predict_transition(self, state, adj=None):
        if not self.episodic_memory.episodes:
            return None
            
        if adj is None:
            if self._latest_adj is None:
                raise RuntimeError("No adjacency available. Pass adj explicitly or call store_regime() first.")
            compressed_current = self._latest_adj
        else:
            compressed_current = self.motif_archive.compress(adj)
            
        best_match_idx = -1
        best_dist = float('inf')
        
        for i, (hist_adj, hist_states) in enumerate(self.episodic_memory.episodes[:-1]):
            dist = self.engine.compute_distance(compressed_current, hist_adj)
            if dist < best_dist:
                best_dist = dist
                best_match_idx = i
                
        if best_match_idx >= 0 and best_dist < 5.0:
            next_adj, _ = self.episodic_memory.episodes[best_match_idx + 1]
            return {
                "predicted_next_adj": next_adj,
                "historical_match_cost": best_dist
            }
            
        return None

    def retrieve_similar(self, state, adj=None, top_k=3, metric="euclidean"):
        if not self.episodic_memory.episodes:
            return []
            
        if hasattr(state, "detach"):
            query = state.detach().cpu().numpy().flatten()
        else:
            query = np.asarray(state).flatten()
            
        results = []
        for i, (hist_adj, hist_state) in enumerate(self.episodic_memory.episodes):
            if hasattr(hist_state, "detach"):
                h_val = hist_state.detach().cpu().numpy().flatten()
            else:
                h_val = np.asarray(hist_state).flatten()
                
            if metric == "euclidean":
                score = -float(np.linalg.norm(query - h_val))
            elif metric == "cosine":
                norm_q = np.linalg.norm(query)
                norm_h = np.linalg.norm(h_val)
                if norm_q == 0 or norm_h == 0:
                    score = 0.0
                else:
                    score = float(np.dot(query, h_val) / (norm_q * norm_h))
            else:
                score = -float(np.linalg.norm(query - h_val))
                
            results.append({
                "similarity_score": score,
                "adj": hist_adj,
                "state": hist_state,
                "metadata": self._metadata[i] if i < len(self._metadata) else None
            })
            
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
