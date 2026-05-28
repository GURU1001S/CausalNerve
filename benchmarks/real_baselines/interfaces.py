from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np

class CausalBaselineInterface(ABC):
    """
    Unified interface for all causal discovery baselines to ensure
    honest, reproducible, and standardized benchmarking.
    """
    
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.params = kwargs
        self.runtime_sec = 0.0
        self.is_fitted = False
        self._adj_matrix = None
        self._confidence_matrix = None

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Fits the causal discovery algorithm to the time-series data.
        Data format: (T, N) where T=time_steps, N=variables.
        """
        pass

    @abstractmethod
    def predict_structure(self) -> np.ndarray:
        """
        Returns the binary or weighted adjacency matrix (N, N).
        """
        pass

    @abstractmethod
    def confidence_scores(self) -> np.ndarray:
        """
        Returns a (N, N) matrix of confidence scores or p-values.
        """
        pass
        
    def runtime_metrics(self) -> Dict[str, float]:
        """Returns computational profiling metrics."""
        return {"total_fit_time_seconds": self.runtime_sec}
