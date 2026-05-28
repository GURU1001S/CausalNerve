import sys
from causalnerve import CausalNerve
from causalnerve_observe.dashboard import CausalRuntimeObservatory

def main():
    print("Initializing CausalNerve...")
    # Initialize a dummy engine with some nodes for the dashboard UI
    nerve = CausalNerve(nodes=10, state_dim=32)
    nerve.preset_name = "Dashboard Demo (MSRB Mock)"
    nerve.current_cycle = 500
    
    # Manually attach the replay engine for the demo
    from causalnerve.memory.replay_engine import StructuralReplayEngine
    nerve.replay_engine = StructuralReplayEngine()
    
    # Pre-populate some history into the replay engine so we can test the scrubber
    for cycle in range(0, 501, 10):
        # Create a simple ring topology
        adjacency = [[i, (i+1)%10, 0.5] for i in range(10)]
        if cycle > 250:
            adjacency.append([2, 5, 0.8]) # add a new edge later in time
            
        nerve.replay_engine.record_snapshot(
            cycle=cycle,
            adjacency_matrix=adjacency,
            leakage=0.05 + (cycle/10000.0),
            v_energy=5.0 - (cycle/1000.0),
            active_alarms=[2] if cycle % 50 == 0 else [],
            node_labels=getattr(nerve, 'node_labels', {i: f'Node {i}' for i in range(10)})
        )
    
    print("Launching Universal Dashboard...")
    dashboard = CausalRuntimeObservatory(nerve)
    
    # Launching on port 7860
    dashboard.launch(port=7860)

if __name__ == "__main__":
    main()
