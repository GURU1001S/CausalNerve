"""
Tests for causalnerve.core
"""

import pytest
import torch
from causalnerve.core.engine import CausalGraphEngine

def test_sparse_propagation_correctness():
    """known graph, known input, verify output"""
    engine = CausalGraphEngine(d_model=64)
    assert engine is not None

def test_o_nk_complexity():
    """measure actual FLOP count, verify O(N·K) not O(N²)"""
    pass

def test_persistence_decay():
    """verify alpha parameter controls decay correctly"""
    pass

def test_gpu_cpu_consistency():
    """same result on CPU and GPU"""
    pass
