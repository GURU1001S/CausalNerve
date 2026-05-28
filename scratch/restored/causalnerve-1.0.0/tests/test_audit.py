import pytest
import os
from causalnerve.reporting.trail import AuditTrail, EvidencePacket, RootCauseNode
from causalnerve.reporting.exporter import AuditExporter

def test_evidence_packet():
    ev = EvidencePacket(
        leakage_before=0.5,
        leakage_after=0.2,
        confidence_score=0.9,
        counterfactual_divergence=0.1,
        motif_similarity=0.8,
        physical_constraints_triggered=["THERMODYNAMIC_VIOLATION"]
    )
    assert ev.leakage_after == 0.2
    d = ev.to_dict()
    assert "leakage_before" in d

def test_audit_trail_and_tree():
    trail = AuditTrail()
    
    ev = EvidencePacket(0.5, 0.2, 0.9, 0.1, 0.8)
    
    n1 = trail.log_surgery("E1", "REJECT", (0, 1), "Constraint", ev)
    assert n1.event_id == "E1"
    
    n2 = trail.log_surgery("E2", "ACCEPT", (1, 2), "Fix", ev, dependencies=["E1"])
    assert len(trail.history) == 2
    
    chain = trail.tree.get_chain("E2")
    assert len(chain) == 2
    assert chain[0].event_id == "E2"
    assert chain[1].event_id == "E1"

def test_audit_exporter():
    trail = AuditTrail()
    ev = EvidencePacket(0.5, 0.2, 0.9, 0.1, 0.8)
    trail.log_surgery("E1", "ACCEPT", (1, 2), "Fix", ev)
    
    AuditExporter.export_json(trail, "TEST_AUDIT.json")
    AuditExporter.export_markdown(trail, "TEST_AUDIT.md")
    
    assert os.path.exists("TEST_AUDIT.json")
    assert os.path.exists("TEST_AUDIT.md")
    
    os.remove("TEST_AUDIT.json")
    os.remove("TEST_AUDIT.md")
