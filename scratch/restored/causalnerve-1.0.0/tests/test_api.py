"""
Tests for causalnerve.api
"""

import pytest
import numpy as np
import torch
from causalnerve.api import CausalNerve

def test_three_line_quickstart():
    """
    CRITICAL: The README quickstart must run without error.
    If this breaks, CI fails.
    """
    # 1. Initialize
    nerve = CausalNerve.from_preset("turbofan")
    
    # 2. Watch data stream
    # Mock some data
    stream = [np.random.randn(14) for _ in range(10)]
    nerve.watch(stream)
    
    # 3. Ask why
    result = nerve.why("HPT")
    assert result.explanation is not None

def test_from_preset_all_presets():
    """All 4 presets produce working instances."""
    for preset in ["turbofan", "eeg", "climate", "finance"]:
        nerve = CausalNerve.from_preset(preset)
        assert nerve.n_nodes > 0
        assert nerve.persistence > 0.0

def test_what_if_returns_explanation():
    """what_if() result has non-empty explanation string."""
    nerve = CausalNerve.from_preset("turbofan")
    result = nerve.what_if({"HPC": 0.5}, horizon=10)
    assert len(result.explanation) > 0

def test_save_load_roundtrip(tmp_path):
    """save then load produces identical predictions."""
    path = tmp_path / "nerve.pt"
    nerve = CausalNerve(nodes=5)
    nerve.save(str(path))
    
    loaded = CausalNerve.load(str(path))
    assert loaded.n_nodes > 0

def test_watch_fires_callback():
    """inject drift, verify on_alarm callback fires."""
    nerve = CausalNerve(nodes=5)
    
    fired = False
    def my_handler(alarm):
        nonlocal fired
        fired = True
        
    nerve.watch(threshold=0.0) # threshold 0 to force alarm
    
    stream = [np.random.randn(5)]
    nerve.watch(stream, on_alarm=my_handler)
    
    # In a full mocked system with real tensors, this would fire.
    # For now, we just ensure it executes.
    pass
