"""
Messy Streaming Reality Benchmark (MSRB)
A corruption layer for testing causal algorithms under realistic industrial telemetry failure modes.
"""

from .corruption import SensorPathologyInjector
from .temporal import TemporalDisorderInjector
from .human_noise import HumanInterventionNoiseInjector
from .structural import StructuralAmbiguityInjector
from .evaluator import MessyRealityEvaluator
from .suite import MSRBSuite

__all__ = [
    'SensorPathologyInjector',
    'TemporalDisorderInjector',
    'HumanInterventionNoiseInjector',
    'StructuralAmbiguityInjector',
    'MessyRealityEvaluator',
    'MSRBSuite'
]
