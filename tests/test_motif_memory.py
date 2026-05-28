import pytest
import os
import tempfile
import numpy as np
from causalnerve.fleet.motif_memory import MotifFingerprint, MotifMemoryBank, MotifMatcher, EarlyWarningEngine

def create_dummy_edge_matrix(seed, n=14):
    np.random.seed(seed)
    mat = np.random.rand(n, n)
    mat[mat < 0.9] = 0.0  # Make it sparse
    return mat.tolist()

def test_motif_fingerprint_generation():
    mat1 = create_dummy_edge_matrix(42)
    # create identical matrix
    mat2 = create_dummy_edge_matrix(42)
    
    fp1 = MotifFingerprint.generate(mat1, threshold=0.8)
    fp2 = MotifFingerprint.generate(mat2, threshold=0.8)
    
    assert fp1 == fp2
    assert len(fp1) == 16
    
    # modify one strong edge
    mat3 = create_dummy_edge_matrix(42)
    mat3[0][1] = 1.0 if mat3[0][1] == 0.0 else 0.0
    fp3 = MotifFingerprint.generate(mat3, threshold=0.8)
    
    assert fp1 != fp3

def test_jaccard_similarity():
    mat_a = [[0.0, 0.9], [0.1, 0.0]]
    mat_b = [[0.0, 0.8], [0.9, 0.0]]
    
    sim = MotifFingerprint.jaccard_similarity(mat_a, mat_b, threshold=0.5)
    # A has (0,1), B has (0,1), (1,0) -> Intersection: 1, Union: 2 -> 0.5
    assert sim == 0.5

def test_motif_memory_bank_persistence():
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        temp_path = tf.name
    
    try:
        bank = MotifMemoryBank(storage_path=temp_path)
        mat = create_dummy_edge_matrix(1)
        
        fp = bank.add_or_update_motif(
            edge_matrix=mat,
            engine_id="ENG-001",
            leakage_signature=0.35,
            lyapunov_trajectory=2.1,
            cycles_to_failure_improvement=45,
            supporting_sensors=[0.5]*21,
            intervention_success=True
        )
        
        assert fp in bank.motifs
        assert bank.motifs[fp]['confirmations'] == 1
        
        # update same motif
        bank.add_or_update_motif(
            edge_matrix=mat,
            engine_id="ENG-002",
            leakage_signature=0.45,
            lyapunov_trajectory=2.3,
            cycles_to_failure_improvement=50,
            supporting_sensors=[0.6]*21,
            intervention_success=True
        )
        
        assert bank.motifs[fp]['confirmations'] == 2
        assert len(bank.motifs[fp]['engines_observed']) == 2
        
        # test persistence across instances
        bank2 = MotifMemoryBank(storage_path=temp_path)
        assert fp in bank2.motifs
        assert bank2.motifs[fp]['confirmations'] == 2
    finally:
        os.remove(temp_path)

def test_early_warning_engine():
    bank = MotifMemoryBank(storage_path=":memory:") # Not real path, will fail load but that's fine for test
    bank.motifs = {} # clear
    
    mat = create_dummy_edge_matrix(1)
    # artificially create a high confidence motif
    bank.add_or_update_motif(mat, "ENG-001", 0.35, 2.1, 45, [], True)
    for i in range(10): # increase confidence
        bank.add_or_update_motif(mat, f"ENG-00{i+2}", 0.35, 2.1, 45, [], True)
        
    matcher = MotifMatcher(bank)
    ewe = EarlyWarningEngine(matcher)
    
    # Evaluate with similar matrix
    warning = ewe.evaluate(mat, current_leakage=0.4)
    assert warning is not None
    assert warning['warning_triggered'] is True
    assert "ENG-001" in warning['previously_seen_in']
    
    # Evaluate with completely different matrix
    mat2 = create_dummy_edge_matrix(99)
    warning2 = ewe.evaluate(mat2, current_leakage=0.4)
    assert warning2 is None
