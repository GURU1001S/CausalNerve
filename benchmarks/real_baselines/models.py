import time
import numpy as np
from typing import Dict, Any, Optional

from benchmarks.real_baselines.interfaces import CausalBaselineInterface

class MockVarLingam(CausalBaselineInterface):
    """
    Wrapper for VARLiNGAM (via the lingam library).
    Demonstrates integration of a real baseline.
    """
    def __init__(self, seed: int = 42, lags: int = 1, **kwargs):
        super().__init__(seed, **kwargs)
        self.lags = lags
        try:
            import lingam
            self.model = lingam.VARLiNGAM(lags=self.lags)
        except ImportError:
            self.model = None # Stub for environments without lingam
            
    def fit(self, data: np.ndarray) -> None:
        start_time = time.time()
        np.random.seed(self.seed)
        
        if self.model is not None:
            import pandas as pd
            df = pd.DataFrame(data)
            self.model.fit(df)
            # VARLiNGAM produces a list of adjacency matrices for lags
            self._adj_matrix = self.model.adjacency_matrices_[0].T
        else:
            # Fallback stub if library missing
            n = data.shape[1]
            self._adj_matrix = np.random.rand(n, n)
            np.fill_diagonal(self._adj_matrix, 0)
            
        self.runtime_sec = time.time() - start_time
        self.is_fitted = True

    def predict_structure(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        # Thresholding for binary structure
        return (np.abs(self._adj_matrix) > 0.1).astype(float)

    def confidence_scores(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        # Normalize weights as a proxy for confidence if p-values unavailable
        return np.abs(self._adj_matrix) / (np.max(np.abs(self._adj_matrix)) + 1e-9)

class MockDynoTears(CausalBaselineInterface):
    """
    Wrapper for DYNOTEARS (via causalnex).
    """
    def __init__(self, seed: int = 42, p: int = 1, w_threshold: float = 0.01, **kwargs):
        super().__init__(seed, **kwargs)
        self.p = p
        self.w_threshold = w_threshold
        
    def fit(self, data: np.ndarray) -> None:
        start_time = time.time()
        np.random.seed(self.seed)
        try:
            from causalnex.structure.dynotears import from_pandas_dynamic
            import pandas as pd
            df = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
            sm = from_pandas_dynamic(df, p=self.p, w_threshold=self.w_threshold)
            
            n = data.shape[1]
            adj = np.zeros((n, n))
            for u, v, w in sm.edges(data=True):
                # We extract lag=0 edges for the instantaneous DAG
                if "lag0" in u and "lag0" in v:
                    ui = int(u.split("_")[0])
                    vi = int(v.split("_")[0])
                    adj[ui, vi] = w.get("weight", 1.0)
            self._adj_matrix = adj
        except ImportError:
            n = data.shape[1]
            self._adj_matrix = np.random.rand(n, n)
            np.fill_diagonal(self._adj_matrix, 0)
            
        self.runtime_sec = time.time() - start_time
        self.is_fitted = True

    def predict_structure(self) -> np.ndarray:
        return (np.abs(self._adj_matrix) > 0.0).astype(float)

    def confidence_scores(self) -> np.ndarray:
        return np.abs(self._adj_matrix)

class MockPCMCI(CausalBaselineInterface):
    """
    Wrapper for Tigramite's PCMCI.
    """
    def __init__(self, seed: int = 42, tau_max: int = 1, pc_alpha: float = 0.1, **kwargs):
        super().__init__(seed, **kwargs)
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        
    def fit(self, data: np.ndarray) -> None:
        start_time = time.time()
        np.random.seed(self.seed)
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
            
            dataframe = pp.DataFrame(data)
            cond_ind_test = ParCorr(significance='analytic')
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
            results = pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=self.pc_alpha)
            
            # Extract p-matrix for lag 0
            n = data.shape[1]
            p_matrix = results['p_matrix'][:, :, 0]
            val_matrix = results['val_matrix'][:, :, 0]
            
            # Confidence is 1 - p_value
            self._confidence_matrix = 1.0 - p_matrix
            self._adj_matrix = val_matrix
        except ImportError:
            n = data.shape[1]
            self._confidence_matrix = np.random.rand(n, n)
            self._adj_matrix = np.random.rand(n, n)
            
        self.runtime_sec = time.time() - start_time
        self.is_fitted = True

    def predict_structure(self) -> np.ndarray:
        return (self._confidence_matrix > (1.0 - self.pc_alpha)).astype(float)

    def confidence_scores(self) -> np.ndarray:
        return self._confidence_matrix
