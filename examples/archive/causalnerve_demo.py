#!/usr/bin/env python3
"""
CausalNerve Intelligence Observatory — Standalone Demo
NASA C-MAPSS FD004 · No real model required.
Simulates a complete 600-cycle turbofan monitoring scenario.
"""
import time, math, random, sys
import numpy as np

# ---------------------------------------------------------------------------
# Physical model constants
# ---------------------------------------------------------------------------
TRUE_EDGES = [
    (11,3),(3,4),(4,2),(4,6),(6,2),(2,1),(5,7),(7,0),(9,4),(10,1),(4,12),(3,12),
]
IMPOSSIBLE_EDGES = [(13,0),(10,3),(12,4)]
DEGRADATION_CASCADE = [11,3,4,2,1,0]
THERMAL_THRESHOLD = 0.81
N_NODES = 14
SENSORS = [
    "T2","T24","T30","T50","P2","P15","P30",
    "Nf","Nc","epr","Ps30","phi","NRf","NRc",
    "BPR","farB","htBleed","Nf_dmd","PCNfR","W31","W32",
]

# ---------------------------------------------------------------------------
# Synthetic sensor generation (physically plausible)
# ---------------------------------------------------------------------------
def simulate_sensors(cycle, max_cyc=600, noise=0.02):
    degradation = min(cycle / max_cyc, 1.0)
    T30 = 0.5 + 0.4 * degradation + random.gauss(0, noise)
    T50 = 0.4 + 0.35 * degradation + random.gauss(0, noise)
    P30 = 0.6 + 0.25 * degradation + random.gauss(0, noise)
    Nf  = max(0, 0.8 - 0.3 * degradation + random.gauss(0, noise))
    phi = 0.5 + 0.4 * degradation + random.gauss(0, noise)
    return [
        0.52+random.gauss(0,noise), 0.45+0.1*degradation+random.gauss(0,noise),
        T30, T50, 0.55+random.gauss(0,noise), 0.48+random.gauss(0,noise), P30,
        Nf, Nf*0.95+random.gauss(0,noise*0.5), 0.5+0.2*degradation+random.gauss(0,noise),
        P30*0.92+random.gauss(0,noise*0.5), phi,
        Nf*1.1+random.gauss(0,noise*0.5), Nf*1.05+random.gauss(0,noise*0.5),
        max(0,0.6-0.2*degradation+random.gauss(0,noise)),
        phi*0.9+random.gauss(0,noise*0.5), 0.4+0.35*degradation+random.gauss(0,noise),
        Nf+random.gauss(0,noise*0.3), Nf+random.gauss(0,noise*0.3),
        0.3+0.2*degradation+random.gauss(0,noise), 0.25+0.2*degradation+random.gauss(0,noise),
    ]

# ---------------------------------------------------------------------------
# Edge probability simulation
# ---------------------------------------------------------------------------
def simulate_edge_probs(cycle, max_cyc=600):
    ep = np.random.uniform(0, 0.01, (N_NODES, N_NODES))
    np.fill_diagonal(ep, 0)
    progress = min(cycle / max_cyc, 1.0)
    # Discovery schedule
    discovery_schedule = [
        ((11,3), 30), ((3,4), 45), ((5,7), 60), ((7,0), 70),
        ((9,4), 90), ((10,1), 100), ((4,6), 120), ((6,2), 140),
        ((4,12), 139), ((3,12), 155), ((4,2), 187), ((2,1), 220),
    ]
    for (i,j), disc_cyc in discovery_schedule:
        if cycle >= disc_cyc:
            ramp = min((cycle - disc_cyc) / 40.0, 1.0)
            ep[i,j] = 0.3 + 0.65 * ramp + random.gauss(0, 0.02)
    # Impossible edges stay near zero
    for (i,j) in IMPOSSIBLE_EDGES:
        ep[i,j] = max(0, random.gauss(0, 0.005))
    return np.clip(ep, 0, 1)

# ---------------------------------------------------------------------------
# Node state simulation
# ---------------------------------------------------------------------------
def simulate_node_states(cycle, max_cyc=600, world=0):
    degradation = min(cycle / max_cyc, 1.0)
    base = np.full(N_NODES, 0.1)
    cascade_weights = {11:0.3, 3:0.5, 4:0.8, 2:0.7, 1:0.15, 0:0.12}
    for nid, w in cascade_weights.items():
        onset = {11:0, 3:50, 4:100, 2:130, 1:200, 0:250}.get(nid, 0)
        if cycle > onset:
            frac = min((cycle - onset) / (max_cyc - onset + 1), 1.0)
            base[nid] = w * frac
    base[5] = 0.4 * degradation
    base[6] = 0.35 * degradation
    base[9] = 0.15 * degradation
    if world == 1 and cycle >= 187:
        repair_frac = min((cycle - 187) / 80.0, 1.0)
        for nid in [4,2,3,5,6]:
            base[nid] *= (1.0 - 0.6 * repair_frac)
    for i in range(N_NODES):
        base[i] += random.gauss(0, 0.01)
    return np.clip(base, 0, 1).tolist()

# ---------------------------------------------------------------------------
# Leakage simulation
# ---------------------------------------------------------------------------
def simulate_leakage(cycle):
    if cycle < 100:
        L = 0.003 + random.gauss(0, 0.001)
    elif cycle < 130:
        L = 0.003 + 0.015 * ((cycle - 100) / 30.0) + random.gauss(0, 0.002)
    elif cycle < 187:
        L = 0.018 + 0.294 * ((cycle - 130) / 57.0) ** 2 + random.gauss(0, 0.005)
    elif cycle < 250:
        peak = 0.312
        decay = min((cycle - 187) / 63.0, 1.0)
        L = peak * (1 - 0.95 * decay) + random.gauss(0, 0.003)
    else:
        L = 0.008 + random.gauss(0, 0.002)
    return max(0.001, L)

def simulate_leakage_components(L):
    return {
        "residual": round(0.38 * L, 4),
        "cf_inconsistency": round(0.29 * L, 4),
        "topology": round(0.22 * L, 4),
        "entropy": round(0.11 * L, 4),
    }

# ---------------------------------------------------------------------------
# Loss simulation
# ---------------------------------------------------------------------------
def simulate_losses(cycle, max_cyc=600):
    progress = min(cycle / max_cyc, 1.0)
    base = 2.5 * math.exp(-3.0 * progress) + 0.05
    return {
        "loss": base + random.gauss(0, 0.02),
        "dag_loss": 0.5 * base * 0.3 + random.gauss(0, 0.005),
        "sparsity_loss": 0.2 * base + random.gauss(0, 0.008),
        "mcd_loss": 0.15 * base * 0.5 + random.gauss(0, 0.003),
        "med_loss": 0.1 * base * 0.4 + random.gauss(0, 0.003),
    }

# ---------------------------------------------------------------------------
# Phase / architecture state
# ---------------------------------------------------------------------------
def simulate_phase(cycle):
    if cycle < 100:
        return "explore", 1.0, 1.0, min(cycle / 100, 1.0)
    elif cycle < 200:
        alpha = 1.0 + 0.5 * ((cycle - 100) / 100.0)
        return "compete", alpha, max(0.3, 1.0 - 0.7 * ((cycle - 100) / 100.0)), (cycle - 100) / 100.0
    else:
        return "sparsify", 2.0, 0.3, min((cycle - 200) / 200.0, 1.0)

# ---------------------------------------------------------------------------
# Fleet simulation
# ---------------------------------------------------------------------------
def simulate_fleet(cycle):
    engines = {}
    offsets = {"E-004": 23, "E-017": 51, "E-022": 89, "E-030": 120}
    for eid, offset in offsets.items():
        shifted = max(0, cycle - offset)
        health = max(0, 1.0 - shifted / 500.0)
        engines[eid] = round(health + random.gauss(0, 0.02), 3)
    engines["E-011"] = round(max(0, 1.0 - cycle / 500.0), 3)
    return engines

# ---------------------------------------------------------------------------
# OCGR event generation
# ---------------------------------------------------------------------------
def generate_ocgr_events(cycle):
    events = []
    schedule = {
        139: {"edge": [4, 12], "label": "HPT→Snsr.A", "action": "ACCEPT", "confidence": 0.61,
              "rationale": "sensor_thermal_contamination", "V_before": 4.25, "V_after": 3.41,
              "w0_leak": 0.92, "w1_leak": 0.041},
        155: {"edge": [3, 12], "label": "Combustor→Snsr.A", "action": "ACCEPT", "confidence": 0.58,
              "rationale": "combustion_noise_coupling", "V_before": 3.41, "V_after": 3.12,
              "w0_leak": 0.85, "w1_leak": 0.038},
        187: {"edge": [4, 2], "label": "HPT→HPC", "action": "ACCEPT", "confidence": 0.73,
              "rationale": "thermal_feedback_hp_spool", "V_before": 3.41, "V_after": 2.14,
              "w0_leak": 2.513, "w1_leak": 0.031},
        44:  {"edge": [13, 0], "label": "Snsr.B→Fan", "action": "REJECT", "confidence": 0.0,
              "rationale": "THERMO_IMPLAUSIBLE", "V_before": 5.0, "V_after": 5.0,
              "w0_leak": 0.003, "w1_leak": 0.003},
    }
    if cycle in schedule:
        ev = schedule[cycle]
        events.append(ev)
    return events

# ---------------------------------------------------------------------------
# Hypothesis library
# ---------------------------------------------------------------------------
def build_hypotheses(cycle):
    hyps = []
    if cycle >= 44:
        hyps.append({"edge":[13,0],"label":"Snsr.B→Fan","state":"REFUTED","confidence":0.0,
                      "reason":"THERMO_IMPLAUSIBLE","mechanism":"Sensor readout cannot cause turbine physics",
                      "fleet_count":0})
    if cycle >= 139:
        hyps.append({"edge":[4,12],"label":"HPT→Snsr.A","state":"CONFIRMED","confidence":0.61,
                      "reason":"","mechanism":"Thermal stress contaminates sensor","fleet_count":15})
    if cycle >= 155:
        hyps.append({"edge":[3,12],"label":"Combustor→Snsr.A","state":"CONFIRMED","confidence":0.58,
                      "reason":"","mechanism":"Combustion noise in sensor","fleet_count":12})
    if cycle >= 130 and cycle < 187:
        hyps.append({"edge":[4,2],"label":"HPT→HPC","state":"TESTING","confidence":round(0.3+0.43*((cycle-130)/57.0),2),
                      "reason":"","mechanism":"HP spool thermal expansion","fleet_count":15})
    elif cycle >= 187:
        hyps.append({"edge":[4,2],"label":"HPT→HPC","state":"CONFIRMED","confidence":0.73,
                      "reason":"","mechanism":"HP spool thermal expansion","fleet_count":15})
    return hyps

# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------
def main():
    from causalnerve_observatory import CausalNerveObservatory
    from causalnerve.fleet.motif_memory import MotifMemoryBank, MotifMatcher, EarlyWarningEngine
    from causalnerve.adaptation.intervention_memory import LongHorizonEvaluator, InterventionRecord
    from causalnerve.core.constraint_engine import PhysicalConstraintEngine
    from causalnerve.reasoning.causal_abstraction import AbstractionLayer
    from causalnerve.fleet.epidemiology import EpidemiologyEngine
    from causalnerve.reasoning.report_generator import ScientificReportGenerator

    print("=" * 70)
    print("  CausalNerve Intelligence Observatory — Demo Mode")
    print("  NASA C-MAPSS FD004 · Simulated Engine Fleet")
    print("=" * 70)

    obs = CausalNerveObservatory(port=8765, scenario="fd004", auto_open=True)
    obs.start()
    print("[LIVE] Dashboard at http://localhost:8765")

    TOTAL_CYCLES = 600
    CYCLE_DELAY = 0.05  # 50ms per cycle

    # Initialize Motif Memory Layer
    bank = MotifMemoryBank(storage_path=":memory:") # pure memory for demo
    bank.clear()
    matcher = MotifMatcher(bank)
    ewe = EarlyWarningEngine(matcher)

    # Pre-seed memory with a historical HPT failure motif from other engines
    historical_ep = np.zeros((N_NODES, N_NODES))
    historical_ep[4, 2] = 0.95 # HPT -> HPC
    historical_ep[11, 3] = 0.8
    historical_ep[3, 4] = 0.8
    
    bank.add_or_update_motif(
        historical_ep.tolist(), "ENG-001", 0.35, 2.1, 45, [0.85]*21, True
    )
    for i in range(15): # Build up confidence
        bank.add_or_update_motif(historical_ep.tolist(), f"ENG-{i+2:03d}", 0.35, 2.1, 45, [0.85]*21, True)

    evaluator = LongHorizonEvaluator()
    
    # Pre-seed a past successful repair to show survival curves instantly
    past_rec = InterventionRecord(
        "hist_001", "E-011", 50, "2026", (4, 2), None, 
        0.31, 3.4, 0.08, 0.02, 2.1, 0.04,
        leakage_trajectory={10: 0.02, 50: 0.03, 100: 0.04},
        lyapunov_trajectory={10: 2.1, 50: 2.2, 100: 2.3},
        survival_duration=120, rollback_occurrence=False
    )
    evaluator.historical_records.append(past_rec)
    
    # Pre-seed a failed repair
    past_rec2 = InterventionRecord(
        "hist_002", "E-012", 50, "2026", (11, 3), None, 
        0.25, 4.1, 0.08, 0.05, 3.1, 0.04,
        survival_duration=40, rollback_occurrence=True
    )
    evaluator.historical_records.append(past_rec2)
    
    physics_engine = PhysicalConstraintEngine()
    abstraction_layer = AbstractionLayer()
    
    epidemiology_engine = EpidemiologyEngine()
    # Seed fleet history
    epidemiology_engine.graph.log_occurrence("ENG-005", "42d351cd", 10, 0.9)
    epidemiology_engine.graph.log_occurrence("ENG-012", "42d351cd", 20, 0.8)
    epidemiology_engine.graph.log_intervention("ENG-005", "42d351cd", (4, 2), True, 15)
    epidemiology_engine.graph.log_intervention("ENG-012", "42d351cd", (4, 2), True, 25)
    epidemiology_engine.graph.log_occurrence("ENG-008", "99a2b1cf", 50, 0.95)

    report_generator = ScientificReportGenerator("E-LIVE")

    prev_div = 0.0
    prev_div_accel = 0.0

    for cycle in range(TOTAL_CYCLES):
        # Simulate all state
        edge_probs = simulate_edge_probs(cycle)
        w0_states = simulate_node_states(cycle, world=0)
        w1_states = simulate_node_states(cycle, world=1)
        leakage = simulate_leakage(cycle)
        leakage_comp = simulate_leakage_components(leakage)
        losses = simulate_losses(cycle)
        phase, alpha, gumbel, phase_prog = simulate_phase(cycle)
        sensors = simulate_sensors(cycle)
        fleet = simulate_fleet(cycle)
        ocgr_events = generate_ocgr_events(cycle)
        hypotheses = build_hypotheses(cycle)

        # Divergence
        w0a = np.array(w0_states)
        w1a = np.array(w1_states)
        divergence = float(np.linalg.norm(w0a - w1a))
        div_accel = divergence - prev_div
        div_accel2 = div_accel - prev_div_accel
        prev_div_accel = div_accel
        prev_div = divergence

        # Lyapunov
        if cycle < 100:
            V = 5.0 - 0.5 * (cycle / 100.0)
        elif cycle < 187:
            V = 4.5 - 1.09 * ((cycle - 100) / 87.0)
        elif cycle < 300:
            V = 3.41 - 1.27 * min((cycle - 187) / 113.0, 1.0)
        else:
            V = 2.14 + random.gauss(0, 0.02)

        # RUL
        rul = max(0, int(400 - cycle * 0.7 + random.gauss(0, 3)))

        # Alarm / warning nodes
        alarm_nodes = [4] if leakage > 0.05 else []
        warning_nodes = [3, 5] if leakage > 0.02 else []
        intervention_nodes = [4, 2] if cycle >= 187 else []

        data = {
            "edge_probs": edge_probs,
            "loss": losses["loss"],
            "dag_loss": losses["dag_loss"],
            "sparsity_loss": losses["sparsity_loss"],
            "mcd_loss": losses["mcd_loss"],
            "med_loss": losses["med_loss"],
            "alpha_t": alpha,
            "gumbel_temp": gumbel,
            "phase": phase,
            "phase_progress": phase_prog,
            "world0_states": w0_states,
            "world1_states": w1_states,
            "divergence": divergence,
            "divergence_acceleration": div_accel2,
            "lyapunov_V": V,
            "leakage_L": leakage,
            "leakage_components": leakage_comp,
            "ocgr_events": ocgr_events,
            "sensor_values": sensors,
            "fleet_states": fleet,
            "rul_prediction": rul,
            "alarm_nodes": alarm_nodes,
            "warning_nodes": warning_nodes,
            "intervention_nodes": intervention_nodes,
            "hypotheses": hypotheses,
            "w0_leakage": leakage * 3.0 if cycle >= 140 else leakage,
            "w1_leakage": leakage * 0.1 if cycle >= 187 else leakage,
            "w0_failure": 231,
            "w1_stable": 203,
        }

        # Evaluate some edges through the physics engine to generate data
        if cycle % 10 == 0:
            physics_engine.evaluate_edge(1, 2, 0.9) # Valid
            physics_engine.evaluate_edge(4, 1, 0.6) # Invalid
        if cycle == 139:
            physics_engine.evaluate_edge(13, 0, 0.8) # Invalid Snsr.B -> Fan

        # Motif Memory Evaluation
        matches = matcher.find_matches(edge_probs.tolist(), threshold=0.5)
        warning = ewe.evaluate(edge_probs.tolist(), leakage)
        
        data["active_motifs"] = matches[:3] # top 3
        data["early_warning"] = warning

        # Long Horizon Evaluation
        if cycle == 187: # The cycle where repair happens in demo
            live_rec = InterventionRecord(
                "live_001", "E-LIVE", 187, "2026", (4, 2), None,
                leakage, V, 0.1, 0.03, 2.14, 0.05
            )
            evaluator.log_intervention(live_rec)
            
        evaluator.update_trajectories(cycle, leakage, V, 0.05)
        data["intervention_metrics"] = evaluator.get_dashboard_metrics()
        data["physics_metrics"] = physics_engine.get_dashboard_metrics()
        data["abstraction"] = abstraction_layer.process(edge_probs.tolist(), threshold=0.5)
        
        epidemiology_engine.process_live_telemetry("E-LIVE", cycle, matches)
        data["epidemiology"] = epidemiology_engine.get_dashboard_metrics("E-LIVE")
        
        # Insert OCGR data for report generator (simulate accepted edge event)
        if cycle in [139, 155, 187]:
            evs = generate_ocgr_events(cycle)
            if evs:
                data["ocgr_chain"] = evs[0]

        obs.update(cycle, data)
        report_generator.log_cycle(cycle, data)

        # Progress logging
        if cycle % 50 == 0 or cycle in [130, 140, 187]:
            tag = "ALARM" if leakage > 0.05 else ("WARN" if leakage > 0.02 else "OK")
            print(f"  [c.{cycle:03d}] {tag:5s}  L={leakage:.4f}  V={V:.2f}  D={divergence:.3f}  phase={phase}")

        time.sleep(CYCLE_DELAY)

    print("\n" + "=" * 70)
    print("  Demo complete. Dashboard remains live at http://localhost:8765")
    report_generator.generate_report()
    print("  Press Ctrl+C to exit.")
    print("=" * 70)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down. Generating final report if aborted early...")
        report_generator.generate_report()

if __name__ == "__main__":
    main()
