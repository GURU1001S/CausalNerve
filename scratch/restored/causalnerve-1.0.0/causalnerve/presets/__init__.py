"""
causalnerve.presets
===================
Domain-specific knowledge presets for CausalNerve.
"""

from .base import CausalPreset
from .turbofan import TurbofanPreset
from .eeg import EEGPreset
from .climate import ClimatePreset
from .finance import FinancePreset

PRESETS = {
    "turbofan": TurbofanPreset,
    "eeg": EEGPreset,
    "climate": ClimatePreset,
    "finance": FinancePreset,
}

def from_preset(name: str) -> CausalPreset:
    """
    Load a domain preset by name.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. "
            f"Available: {list(PRESETS.keys())}. "
            f"Create custom preset by subclassing CausalPreset."
        )
    return PRESETS[name]()

def register_preset(name: str, preset_class: type):
    """
    Register a custom domain preset.
    """
    PRESETS[name] = preset_class

__all__ = [
    "CausalPreset",
    "TurbofanPreset",
    "EEGPreset", 
    "ClimatePreset",
    "FinancePreset",
    "from_preset",
    "register_preset"
]
