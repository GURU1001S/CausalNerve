import sys, os, time, csv, tracemalloc
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causalnerve.api import CausalNerve
from causalnerve.datasets.synthetic import SyntheticStreamGenerator

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def benchmark_graph_scaling():
    print("Running graph-size scaling benchmarks...")
    sizes = [10, 20, 50, 100, 250, 500]
    
    os.makedirs("results", exist_ok=True)
    with open("results/runtime_scaling.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Graph_Size", "Init_Time_s", "Memory_MB", "Status"])
        
        for size in sizes:
            set_seed()
            try:
                tracemalloc.start()
                t0 = time.time()
                nerve = CausalNerve(nodes=size, state_dim=16)
                t1 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                init_time = t1 - t0
                mem_mb = peak / 10**6
                
                print(f"Size: {size:4d} | Time: {init_time:.3f}s | Peak Mem: {mem_mb:6.2f}MB")
                # Enforce safe bounds dynamically
                status = "SAFE" if mem_mb < 300 and init_time < 5.0 else "UNSAFE"
                writer.writerow([size, f"{init_time:.4f}", f"{mem_mb:.2f}", status])
            except Exception as e:
                print(f"Size: {size}, FAILED ({e})")
                writer.writerow([size, -1, -1, "FAILED"])

def benchmark_stream_latency():
    print("\nRunning stream latency benchmarks...")
    nodes = 50
    cycles = 500
    
    set_seed()
    nerve = CausalNerve(nodes=nodes, state_dim=16)
    
    latencies = []
    
    with open("results/stream_latency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cycle", "Latency_ms"])
        
        for cycle in range(cycles):
            obs = torch.randn(1, nodes)
            
            t0 = time.perf_counter()
            nerve.step(obs)
            t1 = time.perf_counter()
            
            lat_ms = (t1 - t0) * 1000
            latencies.append(lat_ms)
            writer.writerow([cycle, f"{lat_ms:.4f}"])
            
    print(f"Mean Latency: {np.mean(latencies):.2f}ms | Max: {np.max(latencies):.2f}ms")
    print(f"Throughput: {1000 / np.mean(latencies):.1f} cycles/sec")

if __name__ == "__main__":
    benchmark_graph_scaling()
    benchmark_stream_latency()
