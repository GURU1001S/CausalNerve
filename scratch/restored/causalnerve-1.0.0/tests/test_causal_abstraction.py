import pytest
import numpy as np
from causalnerve.reasoning.causal_abstraction import (
    AbstractionLayer, MotifCompressor, HierarchicalGraphSummarizer,
    TemporalPatternMiner, MacroCausalState
)

def create_edge_matrix(edges, n=14):
    mat = np.zeros((n, n))
    for u, v, w in edges:
        mat[u, v] = w
    return mat.tolist()

def test_motif_compressor():
    # Compressor edges: 0, 1, 2
    mat = create_edge_matrix([(0, 1, 0.9), (1, 2, 0.8)])
    events = MotifCompressor.compress(mat, threshold=0.5)
    
    assert len(events) == 1
    assert events[0].name == "Compressor Stage Cascade"
    assert events[0].confidence == pytest.approx(0.85)

def test_temporal_pattern_miner():
    miner = TemporalPatternMiner(history_size=5)
    
    # Static
    for _ in range(5):
        miner.update(create_edge_matrix([(0, 1, 0.9)]))
    assert not miner.detect_oscillatory_instability()
    
    # Oscillating
    miner = TemporalPatternMiner(history_size=5)
    for i in range(5):
        val = 0.9 if i % 2 == 0 else 0.1
        miner.update(create_edge_matrix([(0, 1, val)]))
        
    assert miner.detect_oscillatory_instability(threshold=0.3) is True

def test_abstraction_layer():
    layer = AbstractionLayer()
    
    # Combustor edge (3, 4)
    mat = create_edge_matrix([(3, 4, 0.9)])
    res = layer.process(mat)
    
    assert res["macro_state"] == MacroCausalState.COMBUSTION_INSTABILITY.value
    assert "Combustion Dynamics Shift" in res["dominant_motif"]
    assert "Instead of tracking" in res["narrative"]
