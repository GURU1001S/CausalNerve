import os
import urllib.request
import zipfile
import numpy as np
from typing import Optional, List
from .base import CausalDataset, CausalDataBundle

class CMAPSSDataset(CausalDataset):
    """
    NASA C-MAPSS turbofan degradation dataset.
    FD001, FD002, FD003, FD004 subsets.
    
    Auto-downloads from NASA data repository if not cached.
    """
    
    DOWNLOAD_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

    
    def __init__(self, subset: str = "FD001", cache_dir: str = r"D:\Games\RP\datasets\cmapss", normalize: bool = True, include_settings: bool = True, download: bool = True):
        self.subset = subset
        self.cache_dir = os.path.expanduser(cache_dir)
        self.normalize = normalize
        self.include_settings = include_settings
        self.download = download
        
        # Delay download until load() is called, but print citation on init
        print(f"\nUsing NASA C-MAPSS dataset. Please cite:\n{self.citation}")
        
    def _ensure_downloaded(self):
        if not self.download:
            return
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
                            print(f"\rDownload progress: {percent:.1f}%", end="")
                        else:
                            print(f"\rDownloaded {readsofar} bytes", end="")
                            
                    urllib.request.urlretrieve(self.DOWNLOAD_URL, zip_path, reporthook=report)
                    print("\nDownload complete.")
                except Exception as e:
                    print(f"\n[ERROR] Download failed: {e}")
                    print(f"Please manually download from {self.DOWNLOAD_URL} and extract to {self.cache_dir}")
                    raise
            
            print("Extracting CMAPSS dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
                
    def load(self) -> CausalDataBundle:
        self._ensure_downloaded()
        data_file = os.path.join(self.cache_dir, f"train_{self.subset}.txt")
        data = np.loadtxt(data_file)
        
        if self.include_settings:
            # Extract 3 settings + 21 sensors (cols 2 to 25)
            X = data[:, 2:26]
        else:
            # Extract 21 sensor columns (cols 5 to 25)
            X = data[:, 5:26]
        
        if self.normalize:
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            X = (X - mins) / ranges
            
        labels = {}
        sensor_names = [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
            "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
            "Nf_dmd", "PCNfR_dmd", "W31", "W32"
        ]
        if self.include_settings:
            for i, name in enumerate(["Altitude", "Mach", "TRA"]): labels[i] = name
            for i, name in enumerate(sensor_names): labels[i+3] = name
            n_features = 24
        else:
            for i, name in enumerate(sensor_names): labels[i] = name
            n_features = 21

            
        return CausalDataBundle(
            X=X,
            graph_hint=None,
            node_labels=labels,
            metadata={"subset": self.subset, "n_features": n_features, "n_engines": len(np.unique(data[:,0]))}
        )
        
    def load_engine(self, engine_id: int) -> CausalDataBundle:
        self._ensure_downloaded()
        data_file = os.path.join(self.cache_dir, f"train_{self.subset}.txt")
        data = np.loadtxt(data_file)
        engine_data = data[data[:, 0] == engine_id]
        
        if self.include_settings:
            X = engine_data[:, 2:26]
        else:
            X = engine_data[:, 5:26]
        
        if self.normalize:
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            X = (X - mins) / ranges
            
        labels = {}
        sensor_names = [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
            "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
            "Nf_dmd", "PCNfR_dmd", "W31", "W32"
        ]
        if self.include_settings:
            for i, name in enumerate(["Altitude", "Mach", "TRA"]): labels[i] = name
            for i, name in enumerate(sensor_names): labels[i+3] = name
        else:
            for i, name in enumerate(sensor_names): labels[i] = name
            
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
        return """
        @inproceedings{saxena2008damage,
          title={Damage propagation modeling for aircraft 
                 engine run-to-failure simulation},
          author={Saxena, Abhinav and Goebel, Kai and Simon, Don 
                  and Eklund, Neil},
          year={2008}
        }
        """
