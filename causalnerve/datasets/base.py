from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class CausalDataBundle:
    X: np.ndarray  # (T, n_features) - time series
    graph_hint: Optional[List[Tuple[int, int]]] = None # known edges
    node_labels: Optional[Dict[int, str]] = None # feature names
    metadata: Dict = None # dataset info

class CausalDataset:
    """Base class for all CausalNerve datasets."""
    
    def load(self) -> CausalDataBundle:
        raise NotImplementedError
    
    def train_test_split(self, test_fraction: float = 0.2) -> Tuple[CausalDataBundle, CausalDataBundle]:
        raise NotImplementedError
    
    @property
    def citation(self) -> str:
        """BibTeX citation for this dataset. Always included."""
        raise NotImplementedError
