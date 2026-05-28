import os

def create_dataset_files():
    os.makedirs("causalnerve/datasets", exist_ok=True)
    
    # __init__.py
    init_content = """from .base import CausalDataset, CausalDataBundle
from .cmapss import CMAPSSDataset
from .synthetic import SyntheticStreamGenerator
from .eeg_datasets import EEGDataset
from .finance_datasets import FinanceDataset
"""
    with open("causalnerve/datasets/__init__.py", "w") as f: f.write(init_content)

    # base.py
    base_content = """from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class CausalDataBundle:
    X: np.ndarray  # (T, n_features) - time series
    graph_hint: Optional[List[Tuple[int, int]]] = None # known edges
    node_labels: Optional[Dict[int, str]] = None # feature names
    metadata: Dict = None # dataset info

class CausalDataset:
    \"\"\"Base class for all CausalNerve datasets.\"\"\"
    
    def load(self) -> CausalDataBundle:
        raise NotImplementedError
    
    def train_test_split(self, test_fraction: float = 0.2) -> Tuple[CausalDataBundle, CausalDataBundle]:
        raise NotImplementedError
    
    @property
    def citation(self) -> str:
        \"\"\"BibTeX citation for this dataset. Always included.\"\"\"
        raise NotImplementedError
"""
    with open("causalnerve/datasets/base.py", "w") as f: f.write(base_content)

    # cmapss.py
    cmapss_content = """import os
import urllib.request
import zipfile
import numpy as np
from typing import Optional, List
from .base import CausalDataset, CausalDataBundle

class CMAPSSDataset(CausalDataset):
    \"\"\"
    NASA C-MAPSS turbofan degradation dataset.
    FD001, FD002, FD003, FD004 subsets.
    
    Auto-downloads from NASA data repository if not cached.
    \"\"\"
    
    DOWNLOAD_URL = "https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip"
    
    def __init__(self, subset: str = "FD001", cache_dir: str = "~/.causalnerve/data", normalize: bool = True):
        self.subset = subset
        self.cache_dir = os.path.expanduser(cache_dir)
        self.normalize = normalize
        
        # Delay download until load() is called, but print citation on init
        print(f"\\nUsing NASA C-MAPSS dataset. Please cite:\\n{self.citation}")
        
    def _ensure_downloaded(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        zip_path = os.path.join(self.cache_dir, "CMAPSSData.zip")
        data_file = os.path.join(self.cache_dir, f"train_{self.subset}.txt")
        
        if not os.path.exists(data_file):
            if not os.path.exists(zip_path):
                print(f"Downloading CMAPSS dataset from NASA...")
                try:
                    def report(blocknum, blocksize, totalsize):
                        readsofar = blocknum * blocksize
                        if totalsize > 0:
                            percent = readsofar * 1e2 / totalsize
                            print(f"\\rDownload progress: {percent:.1f}%", end="")
                        else:
                            print(f"\\rDownloaded {readsofar} bytes", end="")
                            
                    urllib.request.urlretrieve(self.DOWNLOAD_URL, zip_path, reporthook=report)
                    print("\\nDownload complete.")
                except Exception as e:
                    print(f"\\n[ERROR] Download failed: {e}")
                    print(f"Please manually download from {self.DOWNLOAD_URL} and extract to {self.cache_dir}")
                    raise
            
            print("Extracting CMAPSS dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
                
    def load(self) -> CausalDataBundle:
        self._ensure_downloaded()
        data_file = os.path.join(self.cache_dir, f"train_{self.subset}.txt")
        data = np.loadtxt(data_file)
        # Extract 21 sensor columns (cols 5 to 25)
        X = data[:, 5:26]
        
        if self.normalize:
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            X = (X - mins) / ranges
            
        labels = {i: f"S{i+1}" for i in range(21)}
        return CausalDataBundle(
            X=X,
            graph_hint=None,
            node_labels=labels,
            metadata={"subset": self.subset, "n_sensors": 21, "n_engines": len(np.unique(data[:,0]))}
        )
        
    def load_engine(self, engine_id: int) -> CausalDataBundle:
        self._ensure_downloaded()
        data_file = os.path.join(self.cache_dir, f"train_{self.subset}.txt")
        data = np.loadtxt(data_file)
        engine_data = data[data[:, 0] == engine_id]
        X = engine_data[:, 5:26]
        
        if self.normalize:
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            X = (X - mins) / ranges
            
        labels = {i: f"S{i+1}" for i in range(21)}
        return CausalDataBundle(X=X, node_labels=labels, metadata={"engine_id": engine_id})
        
    def load_fleet(self, n_engines: Optional[int] = None) -> List[CausalDataBundle]:
        self._ensure_downloaded()
        data_file = os.path.join(self.cache_dir, f"train_{self.subset}.txt")
        data = np.loadtxt(data_file)
        engine_ids = np.unique(data[:, 0])
        if n_engines:
            engine_ids = engine_ids[:n_engines]
        return [self.load_engine(eid) for eid in engine_ids]
    
    @property
    def citation(self) -> str:
        return \"\"\"
        @inproceedings{saxena2008damage,
          title={Damage propagation modeling for aircraft 
                 engine run-to-failure simulation},
          author={Saxena, Abhinav and Goebel, Kai and Simon, Don 
                  and Eklund, Neil},
          year={2008}
        }
        \"\"\"
"""
    with open("causalnerve/datasets/cmapss.py", "w") as f: f.write(cmapss_content)

    # synthetic.py
    synthetic_content = """import numpy as np
from typing import Iterator, Tuple

class SyntheticStreamGenerator:
    \"\"\"
    Streaming synthetic data generator for demos and testing.
    Generates data on-the-fly — no file loading needed.
    
    This is what the README quickstart uses.
    Zero external dependencies.
    \"\"\"
    
    @staticmethod
    def stable_system(n_nodes: int = 6, n_cycles: int = 500, seed: int = 42) -> Iterator[np.ndarray]:
        \"\"\"
        Yields observations one cycle at a time.
        Suitable for nerve.watch(stream) usage.
        \"\"\"
        np.random.seed(seed)
        state = np.zeros(n_nodes)
        
        adj = np.zeros((n_nodes, n_nodes))
        edges = [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5)]
        for (u, v) in edges:
            adj[u, v] = 0.5
            
        for _ in range(n_cycles):
            noise = np.random.normal(0, 0.1, n_nodes)
            state = adj.T @ state + noise
            yield state.copy()
            
    @staticmethod
    def with_structural_drift(n_nodes: int = 6, drift_at_cycle: int = 200, new_edge: Tuple[int, int] = (4, 2), seed: int = 42) -> Iterator[np.ndarray]:
        \"\"\"
        Stable until cycle 200, then adds new_edge.
        Used for the flagship demo and quickstart.
        \"\"\"
        np.random.seed(seed)
        state = np.zeros(n_nodes)
        
        adj = np.zeros((n_nodes, n_nodes))
        edges = [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5)]
        for (u, v) in edges:
            adj[u, v] = 0.6
            
        cycle = 0
        while True:
            if cycle == drift_at_cycle:
                adj[new_edge[0], new_edge[1]] = 0.7
                
            noise = np.random.normal(0, 0.1, n_nodes)
            state = adj.T @ state + noise
            yield state.copy()
            cycle += 1
"""
    with open("causalnerve/datasets/synthetic.py", "w") as f: f.write(synthetic_content)

    # eeg_datasets.py
    eeg_content = """from .base import CausalDataset

class EEGDataset(CausalDataset):
    \"\"\"Stub for EEG dataset adapter.\"\"\"
    @property
    def citation(self) -> str:
        return "@article{eeg_stub, title={EEG dataset stub}, year={2026}}"
"""
    with open("causalnerve/datasets/eeg_datasets.py", "w") as f: f.write(eeg_content)

    # finance_datasets.py
    finance_content = """from .base import CausalDataset

class FinanceDataset(CausalDataset):
    \"\"\"Stub for Finance dataset adapter.\"\"\"
    @property
    def citation(self) -> str:
        return "@article{finance_stub, title={Finance dataset stub}, year={2026}}"
"""
    with open("causalnerve/datasets/finance_datasets.py", "w") as f: f.write(finance_content)
    
    print("Dataset adapters built.")

if __name__ == "__main__":
    create_dataset_files()
