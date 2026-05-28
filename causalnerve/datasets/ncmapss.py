import h5py
import numpy as np
from typing import Optional, List, Dict, Any
from .base import CausalDataset, CausalDataBundle

class NCMAPSSDataset(CausalDataset):
    """
    NASA N-CMAPSS (2021) turbofan dataset.
    Reads from local HDF5 file.
    Combines X_s (sensors), X_v (virtual sensors), and W (operating conditions).
    Total nodes = 14 + 14 + 4 = 32 nodes.
    """
    
    def __init__(self, h5_path: str, normalize: bool = True):
        self.h5_path = h5_path
        self.normalize = normalize
        print(f"\nUsing NASA N-CMAPSS dataset at {h5_path}. Please cite:\n{self.citation}")
        
    def load(self) -> CausalDataBundle:
        return self.load_engine(1)

    def load_engine(self, engine_id: int, decimation: int = 50) -> CausalDataBundle:
        with h5py.File(self.h5_path, 'r') as f:
            A = f['A_dev'][:]
            # Column 0 is unit number
            unit_mask = A[:, 0] == engine_id
            
            # Apply decimation to reduce memory and speed up streaming (N-CMAPSS is 1Hz continuous)
            mask_indices = np.where(unit_mask)[0][::decimation]
            
            Xs = f['X_s_dev'][mask_indices]
            Xv = f['X_v_dev'][mask_indices]
            W = f['W_dev'][mask_indices]
            
            X = np.hstack([W, Xs, Xv])
            
            if self.normalize:
                mins = X.min(axis=0)
                maxs = X.max(axis=0)
                ranges = maxs - mins
                ranges[ranges == 0] = 1.0
                X = (X - mins) / ranges
                
            labels = {}
            for i in range(4): labels[i] = f"W{i+1}"
            for i in range(14): labels[i+4] = f"Xs{i+1}"
            for i in range(14): labels[i+18] = f"Xv{i+1}"
                
            return CausalDataBundle(X=X, node_labels=labels, metadata={"engine_id": engine_id, "decimation": decimation})
            
    def load_fleet(self, n_engines: Optional[int] = None) -> List[CausalDataBundle]:
        with h5py.File(self.h5_path, 'r') as f:
            A = f['A_dev'][:]
            engine_ids = np.unique(A[:, 0])
        
        if n_engines:
            engine_ids = engine_ids[:n_engines]
            
        return [self.load_engine(eid) for eid in engine_ids]
        
    @property
    def citation(self) -> str:
        return """
        @article{arias2021aircraft,
          title={Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics},
          author={Arias Chao, Manuel and Kulkarni, Chetan and Goebel, Kai and Fink, Olga},
          journal={Data},
          volume={6},
          number={1},
          pages={5},
          year={2021},
          publisher={MDPI}
        }
        """
