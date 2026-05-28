import numpy as np

class RecurrenceEngine:
    @staticmethod
    def compute_distance(adj1, adj2):
        if adj1.shape != adj2.shape:
            return float('inf')
        return np.sum(np.abs(adj1 - adj2))
