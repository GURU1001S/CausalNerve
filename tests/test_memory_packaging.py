def test_memory_packaging():
    from causalnerve.memory import StructuralMemoryBank, EpisodicMemory, MotifArchive, RecurrenceEngine
    
    bank = StructuralMemoryBank()
    assert hasattr(bank, "store_regime")
    assert hasattr(bank, "predict_transition")
    
    print("Memory packaging test passed.")
