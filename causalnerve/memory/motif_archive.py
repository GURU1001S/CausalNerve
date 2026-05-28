import numpy as np

class MotifArchive:
    def __init__(self):
        self.motifs = []
        
    def compress(self, adj):
        return np.where(adj > 0.05, adj, 0.0)
