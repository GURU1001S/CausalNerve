import sys
import numpy as np
from causalnerve import CausalNerve
from causalnerve_observe.dashboard import CausalRuntimeObservatory
from causalnerve.datasets.cmapss import CMAPSSDataset
from causalnerve.memory.replay_engine import StructuralReplayEngine

def main():
    print("Downloading/Loading NASA C-MAPSS FD001 Dataset...")
    dataset = CMAPSSDataset(subset="FD001", download=True)
    bundle = dataset.load_engine(engine_id=1)
    
    n_nodes = bundle.X.shape[1]
    total_cycles = bundle.X.shape[0]
    
    print(f"Initializing CausalNerve for Turbofan Engine 1 ({n_nodes} nodes, {total_cycles} cycles)...")
    nerve = CausalNerve(nodes=n_nodes, state_dim=32)
    nerve.preset_name = "NASA C-MAPSS FD001 (Engine 1)"
    nerve.current_cycle = total_cycles
    nerve.node_labels = bundle.node_labels
    
    nerve.replay_engine = StructuralReplayEngine()
    
    # Mock some realistic structural discovery over the CMAPSS flight cycle
    # HPT (high pressure turbine) is typically node index 6 or 7, etc.
    print("Simulating causal structural discovery over flight envelope...")
    for cycle in range(0, total_cycles, 10):
        # Build a synthetic adjacency based on real labels
        # e.g., altitude/mach -> sensors
        adjacency = []
        for i in range(3): # Operating conditions affect all sensors
            for j in range(3, n_nodes):
                if np.random.rand() < 0.1: # 10% sparsity
                    adjacency.append([i, j, 0.4])
                    
        # Add some core thermodynamic engine correlations (fake)
        adjacency.append([3, 4, 0.9])
        adjacency.append([6, 7, 0.85])
        adjacency.append([11, 14, 0.7])
        
        # Simulate degradation: add a spurious edge near the end of life
        if cycle > total_cycles * 0.8:
            adjacency.append([7, 18, 0.95])
            
        nerve.replay_engine.record_snapshot(
            cycle=cycle,
            adjacency_matrix=adjacency,
            leakage=0.01 + (cycle / total_cycles) * 0.3, # Leakage rises as engine degrades
            v_energy=12.0 - (cycle / total_cycles) * 4.0, # Energy settles
            active_alarms=[7, 18] if cycle > total_cycles * 0.85 else [],
            node_labels=nerve.node_labels
        )
    
    print("Launching Universal Dashboard...")
    dashboard = CausalRuntimeObservatory(nerve)
    dashboard.launch(port=7860)

if __name__ == "__main__":
    main()
