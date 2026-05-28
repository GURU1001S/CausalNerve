from .episodic_memory import EpisodicMemory
from .motif_archive import MotifArchive
from .recurrence_engine import RecurrenceEngine

class StructuralMemoryBank:
    def __init__(self):
        self.episodic_memory = EpisodicMemory()
        self.motif_archive = MotifArchive()
        self.engine = RecurrenceEngine()
        
    def store_regime(self, adj, states):
        compressed = self.motif_archive.compress(adj)
        self.episodic_memory.add_episode(compressed, states)
        
    def predict_transition(self, current_adj, current_states):
        if not self.episodic_memory.episodes:
            return None
            
        best_match_idx = -1
        best_dist = float('inf')
        
        compressed_current = self.motif_archive.compress(current_adj)
        
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
