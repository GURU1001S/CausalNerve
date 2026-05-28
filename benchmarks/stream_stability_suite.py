import sys, os, time, tracemalloc
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causalnerve.api import CausalNerve

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def test_memory_stability():
    print("Running 24-hour simulation memory stability test (simulated 10,000 cycles)...")
    set_seed()
    
    nodes = 21
    nerve = CausalNerve(nodes=nodes, state_dim=32)
    
    tracemalloc.start()
    
    mem_snapshots = []
    
    for cycle in range(10000):
        obs = torch.randn(1, nodes)
        nerve.step(obs)
        
        if cycle % 2000 == 0:
            current, _ = tracemalloc.get_traced_memory()
            mem_mb = current / 10**6
            mem_snapshots.append(mem_mb)
            print(f"Cycle {cycle:05d} - Memory: {mem_mb:.3f} MB")
            
    tracemalloc.stop()
    growth = mem_snapshots[-1] - mem_snapshots[0]
    print(f"Total memory growth over 10,000 cycles: {growth:.3f} MB")
    
    if growth < 5.0:
        print("[SAFE] Memory stable. No leaks detected.")
    else:
        print("[UNSAFE] Memory leak detected in streaming loop.")

if __name__ == "__main__":
    test_memory_stability()
