import pytest
from causalnerve.runtime.adaptation.calibrator import OnlineCalibrator

def test_online_calibrator():
    cal = OnlineCalibrator(window_size=100)
    
    # Not enough data yet
    cal.update(0.8, 1)
    cal.recalibrate()
    assert cal.is_calibrated == False
    assert cal.calibrate(0.8) == 0.8
    
    # Add enough data for calibration
    for i in range(20):
        # High confidence, usually correct
        cal.update(0.9, 1)
        # Low confidence, usually incorrect
        cal.update(0.2, 0)
        
    cal.recalibrate()
    
    # If sklearn is installed, it will be calibrated. If not, it falls back to identity.
    # In both cases, the output should be between 0 and 1.
    val = cal.calibrate(0.9)
    assert 0.0 <= val <= 1.0
    
    ece = cal.compute_ece()
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0

def test_calibrator_identity_fallback():
    # Test fallback behavior when sklearn is not present or throws error
    cal = OnlineCalibrator()
    # Force sklearn flag off to simulate fallback
    import causalnerve.runtime.adaptation.calibrator as cal_mod
    original = cal_mod.HAS_SKLEARN
    cal_mod.HAS_SKLEARN = False
    
    try:
        for i in range(15):
            cal.update(0.8, 1)
        cal.recalibrate()
        assert cal.is_calibrated == False
        assert cal.calibrate(0.5) == 0.5
    finally:
        cal_mod.HAS_SKLEARN = original
