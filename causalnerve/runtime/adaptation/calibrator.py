import numpy as np
import warnings
from collections import deque
from typing import List, Tuple, Optional

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class OnlineCalibrator:
    """
    Maintains a rolling window of confidence scores vs empirical correctness
    to calibrate the causal reasoning engine's uncertainty outputs.
    """
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.is_calibrated = False
        self._isotonic_model = None

    def update(self, conf: float, correct: int):
        """Add a new observation to the rolling window."""
        self.history.append((conf, correct))
        
    def recalibrate(self):
        """Fit the calibration model based on rolling window history."""
        if len(self.history) < 10:
            return # Not enough data
            
        confs = [x[0] for x in self.history]
        corrects = [x[1] for x in self.history]
        
        if HAS_SKLEARN:
            try:
                # IsotonicRegression expects strictly increasing, out_of_bounds='clip' handles bounds
                self._isotonic_model = IsotonicRegression(out_of_bounds='clip')
                self._isotonic_model.fit(confs, corrects)
                self.is_calibrated = True
            except Exception as e:
                warnings.warn(f"IsotonicRegression failed: {e}. Falling back to identity calibration.")
                self.is_calibrated = False
        else:
            warnings.warn("sklearn not found. OnlineCalibrator falling back to identity calibration.")
            self.is_calibrated = False

    def calibrate(self, conf: float) -> float:
        """Apply calibration to a raw confidence score."""
        if self.is_calibrated and self._isotonic_model is not None:
            # isotonic model predict expects an array-like
            calibrated = self._isotonic_model.predict([conf])[0]
            return float(np.clip(calibrated, 0.0, 1.0))
        # Fallback to identity
        return float(np.clip(conf, 0.0, 1.0))

    def compute_ece(self, n_bins: int = 10) -> float:
        """Compute the Expected Calibration Error (ECE) on the current window."""
        if not self.history:
            return 0.0
            
        confs = np.array([x[0] for x in self.history])
        corrects = np.array([x[1] for x in self.history])
        
        ece = 0.0
        for i in range(n_bins):
            low = i / n_bins
            high = (i + 1) / n_bins
            
            # inclusive upper bound for the last bin
            if i == n_bins - 1:
                mask = (confs >= low) & (confs <= high)
            else:
                mask = (confs >= low) & (confs < high)
                
            if np.sum(mask) > 0:
                bin_acc = np.mean(corrects[mask])
                bin_conf = np.mean(confs[mask])
                ece += (np.sum(mask) / len(confs)) * np.abs(bin_acc - bin_conf)
                
        return float(ece)
