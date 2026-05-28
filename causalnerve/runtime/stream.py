import time
from typing import Dict, Any, Iterator, Optional
from causalnerve.datasets.cmapss import CMAPSSDataset
from causalnerve.datasets.ncmapss import NCMAPSSDataset
class LiveCMAPSSStream:
    """
    Streams one engine cycle-by-cycle from CMAPSS dataset.
    Memory efficient and stream-oriented.
    """
    def __init__(self, engine_id: int = 1, subset: str = "FD001", realtime: bool = True, sleep_factor: float = 0.1, loop: bool = False):
        self.engine_id = engine_id
        self.subset = subset
        self.realtime = realtime
        self.sleep_factor = sleep_factor
        self.loop = loop
        
        if subset.endswith('.h5'):
            self.dataset = NCMAPSSDataset(h5_path=subset)
            self.decimation = 50 # Default decimation for high-frequency N-CMAPSS
        else:
            self.dataset = CMAPSSDataset(subset=subset, include_settings=True)
            self.decimation = 1
            
        self.bundle = None
        self.current_cycle = 0
        self._max_cycles = 0
        
    def reset(self):
        self.current_cycle = 0
        if self.bundle is None:
            # We delay loading to avoid large parsing in init
            if isinstance(self.dataset, NCMAPSSDataset):
                self.bundle = self.dataset.load_engine(self.engine_id, decimation=self.decimation)
            else:
                self.bundle = self.dataset.load_engine(self.engine_id)
            self._max_cycles = self.bundle.X.shape[0]
            
    def step(self) -> Optional[Dict[str, Any]]:
        if self.bundle is None:
            self.reset()
            
        if self.current_cycle >= self._max_cycles:
            if self.loop:
                self.current_cycle = 0
            else:
                return None
                
        # Calculate Remaining Useful Life (RUL) as remaining cycles
        rul = float(self._max_cycles - self.current_cycle - 1)
        x_val = self.bundle.X[self.current_cycle]
        
        result = {
            "cycle": self.current_cycle,
            "x": x_val,
            "rul": rul,
            "engine_id": self.engine_id
        }
        
        self.current_cycle += 1
        
        if self.realtime and self.sleep_factor > 0:
            time.sleep(self.sleep_factor)
            
        return result
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        self.reset()
        while True:
            data = self.step()
            if data is None:
                break
            yield data
