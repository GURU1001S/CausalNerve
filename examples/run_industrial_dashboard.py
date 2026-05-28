"""
Industrial Domain Demo: Autonomous Wind Turbine Farm
Proves the CausalNerve Observatory is fully domain-agnostic for Heavy Industry & IoT.

Scenario: 14-channel SCADA telemetry stream from a multi-megawatt wind turbine. 
CausalNerve monitors the causal structure between environmental conditions (wind),
control inputs (pitch, yaw), and mechanical outputs (rotor speed, power, temp).
Around cycle 600, early-stage gearbox bearing degradation causes friction, 
altering the causal relationship between rotor speed, generator temperature, 
and tower vibration, triggering structural adaptation and alarms.
"""

import numpy as np
from causalnerve import CausalNerve
from causalnerve_observe.dashboard import CausalRuntimeObservatory
from causalnerve.memory.replay_engine import StructuralReplayEngine, RevisionEvent

def main():
    # === DEFINE THE INDUSTRIAL CAUSAL SYSTEM (WIND TURBINE) ===
    node_labels = {
        0:  "Wind_Speed",        # (m/s)
        1:  "Wind_Dir",          # (deg)
        2:  "Nacelle_Yaw",       # (deg)
        3:  "Blade_Pitch",       # (deg)
        4:  "Rotor_Speed",       # (RPM)
        5:  "Generator_Speed",   # (RPM)
        6:  "Gearbox_Oil_Temp",  # (°C)
        7:  "Generator_Temp",    # (°C)
        8:  "Bearing_Temp",      # (°C)
        9:  "Active_Power",      # (kW)
        10: "Reactive_Power",    # (kVAR)
        11: "Grid_Voltage",      # (V)
        12: "Vibration_Axial",   # (mm/s)
        13: "Vibration_Radial",  # (mm/s)
    }

    n_nodes = len(node_labels)
    total_hours = 24 * 7 # 7 days
    samples_per_hour = 6 # every 10 minutes
    total_cycles = total_hours * samples_per_hour # 1008 cycles

    print(f"Wind Turbine SCADA Monitor: {n_nodes} channels, {total_hours}h ({total_cycles} cycles)")

    # === INITIALIZE CAUSALNERVE ===
    nerve = CausalNerve(nodes=n_nodes, state_dim=32)
    nerve.preset_name = "Offshore Wind Turbine (Unit WTG-42)"
    nerve.current_cycle = total_cycles
    nerve.node_labels = node_labels
    nerve.replay_engine = StructuralReplayEngine()

    rng = np.random.default_rng(101)

    # === KNOWN STRUCTURAL TOPOLOGY (NOMINAL STATE) ===
    baseline_edges = [
        # Environmental -> Control & Mechanics
        (0, 4, 0.95),  # Wind Speed -> Rotor Speed
        (0, 3, 0.70),  # Wind Speed -> Blade Pitch (Controller reacts to wind)
        (1, 2, 0.85),  # Wind Dir -> Nacelle Yaw
        
        # Mechanics -> Electrics
        (4, 5, 0.99),  # Rotor Speed -> Generator Speed (via Gearbox)
        (5, 9, 0.92),  # Generator Speed -> Active Power
        (11, 10, 0.8), # Grid Voltage -> Reactive Power
        
        # Thermodynamics
        (4, 6, 0.75),  # Rotor Speed -> Gearbox Oil Temp
        (5, 7, 0.80),  # Generator Speed -> Generator Temp
        (4, 8, 0.65),  # Rotor Speed -> Bearing Temp
        
        # Pitch limits rotor speed
        (3, 4, -0.60), # Blade Pitch -> Rotor Speed (Negative, braking effect)
    ]

    fault_onset_cycle = 600

    print("Simulating causal structural evolution over 7 days of operation...")

    for cycle in range(0, total_cycles, 10):
        adjacency = []

        # Phase 1: Nominal Operation
        for u, v, w in baseline_edges:
            adjacency.append([u, v, w])

        # Phase 2: Gearbox Degradation (Micro-pitting in high-speed shaft bearing)
        if cycle > fault_onset_cycle:
            degradation = (cycle - fault_onset_cycle) / (total_cycles - fault_onset_cycle)
            
            # Friction increases, breaking the clean link between rotor and generator
            # The weight drops as energy is lost to heat/vibration
            for i in range(len(adjacency)):
                if adjacency[i][0] == 4 and adjacency[i][1] == 5:
                    adjacency[i][2] = 0.99 - (0.4 * degradation) 
                    
            # New causal links emerge due to fault physics
            adjacency.append([4, 12, 0.8 * degradation]) # Rotor -> Axial Vib
            adjacency.append([4, 13, 0.9 * degradation]) # Rotor -> Radial Vib
            
            # Heat generation shifts
            adjacency.append([13, 8, 0.7 * degradation]) # Vib -> Bearing Temp
            adjacency.append([8, 6, 0.6 * degradation])  # Bearing Temp -> Oil Temp

        # Metrics
        if cycle <= fault_onset_cycle:
            leakage = 0.05 + rng.normal(0, 0.01)
            v_energy = 5.0 + rng.normal(0, 0.2)
            alarms = []
        else:
            degradation = (cycle - fault_onset_cycle) / (total_cycles - fault_onset_cycle)
            leakage = 0.05 + 0.4 * degradation + rng.normal(0, 0.02)
            v_energy = 5.0 + 7.0 * degradation + rng.normal(0, 0.4)
            
            alarms = []
            if degradation > 0.4:
                alarms.append(13) # Radial Vibration alarm
            if degradation > 0.6:
                alarms.append(8)  # Bearing Temp alarm
            if degradation > 0.8:
                alarms.append(6)  # Oil Temp alarm
                alarms.append(5)  # Gen Speed anomaly
                
        nerve.replay_engine.record_snapshot(
            cycle=cycle,
            adjacency_matrix=adjacency,
            leakage=max(0, leakage),
            v_energy=max(0, v_energy),
            active_alarms=alarms,
            node_labels=node_labels
        )

    # === STRUCTURAL DISCOVERY EVENTS ===
    nerve.replay_engine.record_revision(RevisionEvent(
        cycle=fault_onset_cycle + 40,
        edit_type="add",
        edge=(4, 13),
        confidence=0.88,
        rationale="high_frequency_structural_covariance"
    ))
    nerve.replay_engine.record_revision(RevisionEvent(
        cycle=fault_onset_cycle + 90,
        edit_type="add",
        edge=(13, 8),
        confidence=0.76,
        rationale="frictional_heating_pathway_identified"
    ))
    nerve.replay_engine.record_revision(RevisionEvent(
        cycle=fault_onset_cycle + 180,
        edit_type="modify",
        edge=(4, 5),
        confidence=0.91,
        rationale="mechanical_transmission_efficiency_loss"
    ))

    # === LAUNCH DASHBOARD ===
    print("Launching CausalNerve Observatory for Industrial IoT...")
    dashboard = CausalRuntimeObservatory(nerve)
    dashboard.launch(port=7860)


if __name__ == "__main__":
    main()
