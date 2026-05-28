class EpisodicMemory:
    def __init__(self):
        self.episodes = []
        
    def add_episode(self, adj, states):
        self.episodes.append((adj, states))
