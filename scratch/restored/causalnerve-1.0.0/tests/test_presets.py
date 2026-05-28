"""
Tests for causalnerve.presets
"""
import pytest
from causalnerve.config import from_preset

def test_preset_loading():
    p = from_preset("turbofan")
    assert p.n_nodes == 14
