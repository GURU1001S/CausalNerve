import pytest
import os
import tempfile
from causalnerve.adaptation.intervention_memory import (
    InterventionRecord, LongHorizonEvaluator, InterventionScore,
    SurvivalAnalysis, AuditLayer
)

def test_long_horizon_tracking():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tf:
        temp_path = tf.name
    
    try:
        evaluator = LongHorizonEvaluator(storage_path=temp_path)
        
        # Log an intervention at cycle 100
        record = InterventionRecord(
            intervention_id="int_001", engine_id="E-001", cycle_applied=100, timestamp="2026",
            proposed_edge=(4, 2), removed_edge=None,
            leakage_before=0.5, lyapunov_before=4.0, ece_before=0.1,
            leakage_after=0.05, lyapunov_after=3.0, ece_after=0.05
        )
        evaluator.log_intervention(record)
        
        # Update at cycle 110 (elapsed 10)
        evaluator.update_trajectories(110, leakage=0.06, lyapunov=3.1, ece=0.05)
        
        assert 10 in record.leakage_trajectory
        assert record.leakage_trajectory[10] == 0.06
        assert record.survival_duration == 10
        assert not record.rollback_occurrence
        
        # Test failure condition (leakage spikes)
        # Update at cycle 115 (elapsed 15), leakage spikes > 0.5 * 1.5 = 0.75
        evaluator.update_trajectories(115, leakage=0.8, lyapunov=4.5, ece=0.1)
        assert record.rollback_occurrence is True
        
        # It should have been moved to historical records and exported
        assert len(evaluator.active_records) == 0
        assert len(evaluator.historical_records) == 1
        
        # Verify CSV export occurred
        assert os.path.exists(temp_path)
        with open(temp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 2 # header + 1 row
    finally:
        os.remove(temp_path)

def test_intervention_scoring():
    record = InterventionRecord(
        intervention_id="int_001", engine_id="E-001", cycle_applied=100, timestamp="2026",
        proposed_edge=(4, 2), removed_edge=None,
        leakage_before=0.5, lyapunov_before=4.0, ece_before=0.1,
        leakage_after=0.05, lyapunov_after=3.0, ece_after=0.05,
        leakage_trajectory={10: 0.06, 50: 0.08, 100: 0.1, 150: 0.8}, # breached at 150
        lyapunov_trajectory={10: 3.1, 50: 3.2, 100: 3.3, 150: 4.5},
        survival_duration=150
    )
    
    persistence = InterventionScore.compute_causal_persistence(record)
    assert persistence == 0.75 # 3 out of 4 points are < 0.5
    
    stability = InterventionScore.compute_structural_stability_gain(record)
    assert stability == pytest.approx(1.9)
    
    reward = InterventionScore.compute_delayed_reward(record)
    assert reward > 0

def test_survival_analysis():
    r1 = InterventionRecord(intervention_id="1", engine_id="E", cycle_applied=0, timestamp="", proposed_edge=(0,0), removed_edge=None, leakage_before=0, lyapunov_before=0, ece_before=0, leakage_after=0, lyapunov_after=0, ece_after=0)
    r1.survival_duration = 50
    r1.rollback_occurrence = False
    
    r2 = InterventionRecord(intervention_id="2", engine_id="E", cycle_applied=0, timestamp="", proposed_edge=(0,0), removed_edge=None, leakage_before=0, lyapunov_before=0, ece_before=0, leakage_after=0, lyapunov_after=0, ece_after=0)
    r2.survival_duration = 20
    r2.rollback_occurrence = True # failed at 20
    
    r3 = InterventionRecord(intervention_id="3", engine_id="E", cycle_applied=0, timestamp="", proposed_edge=(0,0), removed_edge=None, leakage_before=0, lyapunov_before=0, ece_before=0, leakage_after=0, lyapunov_after=0, ece_after=0)
    r3.survival_duration = 100
    r3.rollback_occurrence = True # failed at 100
    
    curve = SurvivalAnalysis.kaplan_meier_estimate([r1, r2, r3])
    
    assert 20 in curve
    assert curve[20] == pytest.approx(2/3) # 1 failed out of 3 at risk
    assert curve[100] == 0.0 # 1 failed out of 1 at risk

def test_audit_layer():
    record = InterventionRecord(
        intervention_id="1", engine_id="E", cycle_applied=100, timestamp="", proposed_edge=(4,2), removed_edge=None,
        leakage_before=0.5, lyapunov_before=4.0, ece_before=0.1,
        leakage_after=0.05, lyapunov_after=3.0, ece_after=0.05,
        survival_duration=150, rollback_occurrence=False
    )
    
    text = AuditLayer.generate_explanation(record)
    assert "extended stable operation by +150 cycles" in text
    assert "0.450 reduction" in text
