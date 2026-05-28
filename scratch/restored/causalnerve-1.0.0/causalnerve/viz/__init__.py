from .graph_viz import plot_living_graph, create_timeline_replay
from .counterfactual_viz import plot_counterfactual_split
from .fleet_viz import plot_fleet_recurrence

def plot_worlds(what_if_result, node_labels=None):
    if not node_labels:
        node_labels = {}
    return plot_counterfactual_split(what_if_result, node_labels)

def plot_graph(graph_engine, node_labels, alarms=None, new_edges=None, severed_edges=None):
    if alarms is None: alarms = []
    if new_edges is None: new_edges = []
    if severed_edges is None: severed_edges = []
    
    try:
        adj = graph_engine.get_dense_adjacency()
    except AttributeError:
        # Fallback if engine is mocked
        import numpy as np
        adj = np.zeros((max(node_labels.keys())+1, max(node_labels.keys())+1))
        
    return plot_living_graph(adj, node_labels, alarms, new_edges, severed_edges)

def animate_graph(*args, **kwargs):
    pass
