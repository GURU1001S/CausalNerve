import pytest
from causalnerve.core.constraint_engine import PhysicalConstraintEngine, ConstraintType

def test_sensor_implausibility():
    engine = PhysicalConstraintEngine()
    # Node 13 is Snsr.B, Node 0 is Fan
    res = engine.evaluate_edge(13, 0, 0.95)
    
    assert res.is_valid is False
    assert ConstraintType.SENSOR_IMPLAUSIBILITY.value in res.violations
    assert ConstraintType.DOMAIN_FORBIDDEN.value in res.violations # Snsr.B -> Fan is also explicitly forbidden
    assert res.score == 0.0
    assert res.confidence == 0.0
    assert "Rejected edge Snsr.B -> Fan" in res.explanation

def test_thermodynamic_direction():
    engine = PhysicalConstraintEngine()
    # Node 4 is HPT (turbine), Node 1 is LPC (compressor)
    res = engine.evaluate_edge(4, 1, 0.8)
    
    assert res.is_valid is False
    assert ConstraintType.THERMODYNAMIC_DIRECTION.value in res.violations
    assert res.score == pytest.approx(0.1)

def test_valid_edge():
    engine = PhysicalConstraintEngine()
    # Node 1 is LPC (compressor), Node 2 is HPC (compressor)
    res = engine.evaluate_edge(1, 2, 0.85)
    
    assert res.is_valid is True
    assert len(res.violations) == 0
    assert res.score == 1.0
    assert res.confidence == 0.85
    assert "passed thermodynamic constraints" in res.explanation

def test_dashboard_metrics():
    engine = PhysicalConstraintEngine()
    engine.evaluate_edge(1, 2, 0.9) # Valid
    engine.evaluate_edge(13, 0, 0.8) # Invalid
    engine.evaluate_edge(4, 1, 0.7) # Invalid
    
    metrics = engine.get_dashboard_metrics()
    assert metrics["total_evaluations"] == 3
    assert metrics["total_violations"] == 2
    assert metrics["satisfaction_score"] == pytest.approx(1/3)
    assert len(metrics["recent_rejections"]) == 2
    assert metrics["recent_rejections"][0]["src"] == "Snsr.B"
