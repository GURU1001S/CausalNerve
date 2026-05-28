"""
Medical Domain Demo: ICU Patient Hemodynamic Monitoring
Proves the CausalNerve Observatory is fully domain-agnostic.

Scenario: 16-channel ICU telemetry stream from a post-cardiac-surgery
patient. CausalNerve discovers causal relationships between vitals
and detects when sepsis-induced hemodynamic instability begins
propagating through the cardiovascular system.

No real patient data is used — this is a physiologically-plausible
synthetic stream designed to test the dashboard's universality.
"""

import numpy as np
from causalnerve import CausalNerve
from causalnerve_observe.dashboard import CausalRuntimeObservatory
from causalnerve.memory.replay_engine import StructuralReplayEngine, RevisionEvent

def main():
    # === DEFINE THE MEDICAL CAUSAL SYSTEM ===
    node_labels = {
        0:  "HR",           # Heart Rate (bpm)
        1:  "MAP",          # Mean Arterial Pressure (mmHg)
        2:  "SpO2",         # Peripheral Oxygen Saturation (%)
        3:  "RR",           # Respiratory Rate (breaths/min)
        4:  "CVP",          # Central Venous Pressure (mmHg)
        5:  "CO",           # Cardiac Output (L/min)
        6:  "SvO2",         # Mixed Venous Oxygen Saturation (%)
        7:  "Lactate",      # Blood Lactate (mmol/L)
        8:  "Temp",         # Core Temperature (°C)
        9:  "WBC",          # White Blood Cell Count (×10³/μL)
        10: "CRP",          # C-Reactive Protein (mg/L)
        11: "Creatinine",   # Serum Creatinine (mg/dL)
        12: "UO",           # Urine Output (mL/hr)
        13: "FiO2",         # Fraction of Inspired Oxygen
        14: "PaO2",         # Arterial Oxygen Partial Pressure
        15: "Vasopressor",  # Vasopressor Dose (mcg/kg/min)
    }

    n_nodes = len(node_labels)
    total_hours = 72  # 72 hours of ICU monitoring
    samples_per_hour = 12  # every 5 minutes
    total_cycles = total_hours * samples_per_hour  # 864 cycles

    print(f"ICU Hemodynamic Monitor: {n_nodes} channels, {total_hours}h ({total_cycles} cycles)")

    # === INITIALIZE CAUSALNERVE ===
    nerve = CausalNerve(nodes=n_nodes, state_dim=32)
    nerve.preset_name = "ICU Hemodynamic Monitor (Post-Cardiac Surgery)"
    nerve.current_cycle = total_cycles
    nerve.node_labels = node_labels
    nerve.replay_engine = StructuralReplayEngine()

    rng = np.random.default_rng(42)

    # === KNOWN PHYSIOLOGICAL CAUSAL STRUCTURE ===
    # These are textbook cardiovascular relationships
    baseline_edges = [
        # Cardiac output is driven by heart rate and preload (CVP)
        (0, 5, 0.9),   # HR → CO
        (4, 5, 0.7),   # CVP → CO

        # CO drives MAP
        (5, 1, 0.85),  # CO → MAP

        # MAP drives organ perfusion → urine output
        (1, 12, 0.8),  # MAP → UO

        # Oxygenation chain
        (13, 14, 0.9), # FiO2 → PaO2
        (14, 2, 0.85), # PaO2 → SpO2
        (5, 6, 0.8),   # CO → SvO2

        # Vasopressor → MAP (therapeutic intervention)
        (15, 1, 0.75), # Vasopressor → MAP

        # Temperature → HR (fever increases heart rate)
        (8, 0, 0.6),   # Temp → HR
    ]

    # === SIMULATE 72 HOURS WITH SEPSIS ONSET AT HOUR 36 ===
    sepsis_onset_cycle = 36 * samples_per_hour  # cycle 432

    print("Simulating causal structural evolution over 72h ICU stay...")

    for cycle in range(0, total_cycles, 10):
        hour = cycle / samples_per_hour
        adjacency = []

        # Phase 1 (0-36h): Stable post-operative recovery
        # Standard cardiovascular causal structure
        for u, v, w in baseline_edges:
            adjacency.append([u, v, w])

        # Phase 2 (36-72h): Sepsis onset — new causal pathways emerge
        if cycle > sepsis_onset_cycle:
            sepsis_progress = (cycle - sepsis_onset_cycle) / (total_cycles - sepsis_onset_cycle)

            # Infection markers begin driving inflammatory cascade
            adjacency.append([8, 9, 0.7 * sepsis_progress])    # Temp → WBC
            adjacency.append([9, 10, 0.8 * sepsis_progress])   # WBC → CRP
            adjacency.append([10, 7, 0.6 * sepsis_progress])   # CRP → Lactate

            # Sepsis causes vasodilation → MAP drops despite vasopressor
            adjacency.append([10, 1, -0.5 * sepsis_progress])  # CRP → MAP (negative)

            # Lactate rise indicates tissue hypoperfusion
            adjacency.append([7, 6, -0.7 * sepsis_progress])   # Lactate → SvO2 (negative)

            # Kidney injury from poor perfusion
            if sepsis_progress > 0.4:
                adjacency.append([7, 11, 0.6])  # Lactate → Creatinine
                adjacency.append([11, 12, -0.5]) # Creatinine → UO (negative)

        # Compute physiological leakage and energy
        if cycle <= sepsis_onset_cycle:
            leakage = 0.02 + rng.normal(0, 0.005)
            v_energy = 8.0 + rng.normal(0, 0.3)
            alarms = []
        else:
            sepsis_progress = (cycle - sepsis_onset_cycle) / (total_cycles - sepsis_onset_cycle)
            leakage = 0.02 + 0.25 * sepsis_progress + rng.normal(0, 0.01)
            v_energy = 8.0 + 4.0 * sepsis_progress + rng.normal(0, 0.5)
            alarms = []
            if sepsis_progress > 0.3:
                alarms.append(7)   # Lactate alarm
            if sepsis_progress > 0.5:
                alarms.append(1)   # MAP alarm
                alarms.append(10)  # CRP alarm
            if sepsis_progress > 0.7:
                alarms.append(11)  # Creatinine alarm
                alarms.append(12)  # UO alarm

        nerve.replay_engine.record_snapshot(
            cycle=cycle,
            adjacency_matrix=adjacency,
            leakage=max(0, leakage),
            v_energy=max(0, v_energy),
            active_alarms=alarms,
            node_labels=node_labels
        )

    # === ADD REVISION EVENTS (structural discoveries) ===
    nerve.replay_engine.record_revision(RevisionEvent(
        cycle=sepsis_onset_cycle + 20,
        edit_type="add",
        edge=(8, 9),
        confidence=0.72,
        rationale="fever_wbc_correlation_detected"
    ))
    nerve.replay_engine.record_revision(RevisionEvent(
        cycle=sepsis_onset_cycle + 80,
        edit_type="add",
        edge=(10, 7),
        confidence=0.81,
        rationale="inflammatory_lactate_pathway_discovered"
    ))
    nerve.replay_engine.record_revision(RevisionEvent(
        cycle=sepsis_onset_cycle + 150,
        edit_type="add",
        edge=(7, 11),
        confidence=0.68,
        rationale="acute_kidney_injury_causal_link"
    ))

    # === LAUNCH DASHBOARD ===
    print("Launching CausalNerve Observatory for ICU monitoring...")
    dashboard = CausalRuntimeObservatory(nerve)
    dashboard.launch(port=7860)


if __name__ == "__main__":
    main()
