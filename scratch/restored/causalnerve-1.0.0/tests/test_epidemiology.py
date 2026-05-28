import pytest
from causalnerve.fleet.epidemiology import (
    FleetEpidemiologyGraph,
    MotifPropagationTracker,
    TransferLearningLayer,
    FleetRiskForecaster,
    EpidemiologyEngine
)

def test_epidemiology_graph():
    graph = FleetEpidemiologyGraph()
    graph.log_occurrence("E-01", "motifA", 100, 0.9)
    graph.log_occurrence("E-02", "motifA", 110, 0.8)
    graph.log_occurrence("E-01", "motifB", 120, 0.7)
    
    assert "motifA" in graph.engine_state["E-01"]
    assert len(graph.motif_occurrences["motifA"]) == 2
    
    # Intervention success removes motif from active state
    graph.log_intervention("E-01", "motifA", (1, 2), True, 130)
    assert "motifA" not in graph.engine_state["E-01"]

def test_propagation_tracker():
    graph = FleetEpidemiologyGraph()
    graph.log_occurrence("E-01", "motifA", 100, 0.9)
    graph.log_occurrence("E-02", "motifA", 110, 0.8)
    graph.log_occurrence("E-03", "motifB", 120, 0.7)
    
    assert MotifPropagationTracker.get_most_contagious(graph) == "motifA"
    
    prev = MotifPropagationTracker.compute_prevalence(graph)
    assert prev["motifA"] == 2/3
    assert prev["motifB"] == 1/3

def test_transfer_learning():
    graph = FleetEpidemiologyGraph()
    graph.log_occurrence("E-LIVE", "motifA", 100, 0.9)
    
    # Other engines had success
    graph.log_intervention("E-02", "motifA", (4, 2), True, 50)
    graph.log_intervention("E-03", "motifA", (4, 2), True, 60)
    graph.log_intervention("E-04", "motifA", (1, 2), True, 70)
    
    recs = TransferLearningLayer.get_recommended_interventions(graph, "E-LIVE")
    assert len(recs) == 1
    assert recs[0]["motif"] == "motifA"
    assert recs[0]["recommended_edge"] == (4, 2)
    assert recs[0]["prior_successes"] == 2

def test_engine_integration():
    engine = EpidemiologyEngine()
    engine.process_live_telemetry("E-LIVE", 100, [{"motif_fingerprint": "motifA", "similarity": 0.8}])
    
    metrics = engine.get_dashboard_metrics("E-LIVE")
    assert metrics["most_contagious"] == "motifA"
    assert metrics["total_tracked_engines"] == 1
    assert metrics["fleet_stability_index"] < 1.0
