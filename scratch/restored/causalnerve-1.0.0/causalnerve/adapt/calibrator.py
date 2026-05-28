"""
causalnerve.adaptation.calibrator
============================
Algorithms to align predicted probabilities with empirical accuracy.
Implements Temperature Scaling, offline Isotonic Regression, and Online Calibration.
"""

import numpy as np
import torch
from typing import Tuple, List
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

class ConfidenceCalibrator:
    """
    Transforms raw model scores into perfectly calibrated probabilities.
    Ensures that when the model predicts 0.9 confidence, it is correct 90% of the time.
    """
    def __init__(self, method: str = "temperature"):
        self.method = method
        self.temperature = 1.0
        self.isotonic_model = None
        self.is_fitted = False
        
    def _nll(self, temp: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood objective for temperature scaling."""
        scaled_logits = logits / temp
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        nll = -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        return nll

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        if self.method == "temperature":
            res = minimize(self._nll, x0=[1.0], args=(val_logits, val_labels), bounds=[(0.1, 10.0)])
            self.temperature = res.x[0]
        elif self.method == "isotonic":
            self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            self.isotonic_model.fit(val_probs, val_labels)
        self.is_fitted = True

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return 1.0 / (1.0 + np.exp(-logits))
        if self.method == "temperature":
            scaled_logits = logits / self.temperature
            return 1.0 / (1.0 + np.exp(-scaled_logits))
        elif self.method == "isotonic":
            probs = 1.0 / (1.0 + np.exp(-logits))
            return self.isotonic_model.predict(probs)
        return 1.0 / (1.0 + np.exp(-logits))

    def expected_calibration_error(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs >= lower) & (probs <= upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece)

class CalibrationState:
    STABLE = "STABLE"
    DRIFTING = "DRIFTING"
    COLLAPSED = "COLLAPSED"
    RECOVERING = "RECOVERING"

class OnlineCalibrator:
    """
    Calibration that adapts to distribution shift.
    Implements V2 Reliability requirements:
    - Dynamic Recalibration Frequency
    - Adaptive Sliding Window Compression
    - Confidence Decay
    - State Machine & Emergency Brake
    """
    
    def __init__(self,
                  max_window_size: int = 200,
                  min_window_size: int = 40,
                  decay_factor: float = 0.95):
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.current_window_size = max_window_size
        self.decay_factor = decay_factor
        
        self.recalibrate_stable = 20
        self.recalibrate_ood = 1
        
        self.history = []
        self.updates_since_cal = 0
        self.model = None
        
        self.state = CalibrationState.STABLE
        self.drift_score = 0.0
        self.alpha_decay = 0.5
        self.dynamic_temp = 1.0
        
    def update(self, confidence: float, outcome: bool, weight: float = 1.0, drift_score: float = 0.0):
        self.drift_score = drift_score
        
        # State Machine Transitions
        current_ece = self.ece()
        if current_ece > 0.15:
            self.state = CalibrationState.COLLAPSED
        elif drift_score > 0.5:
            self.state = CalibrationState.DRIFTING
        elif self.state == CalibrationState.COLLAPSED and current_ece < 0.10:
            self.state = CalibrationState.RECOVERING
        elif self.state in [CalibrationState.RECOVERING, CalibrationState.DRIFTING] and current_ece < 0.08 and drift_score < 0.2:
            self.state = CalibrationState.STABLE
            
        # Feature 1 & 2: Dynamic Frequency & Window Compression
        if self.state in [CalibrationState.DRIFTING, CalibrationState.COLLAPSED]:
            self.current_window_size = max(self.min_window_size, int(self.current_window_size * 0.8))
            recal_freq = self.recalibrate_ood
            self.dynamic_temp = 1.0 + 0.1 * drift_score # Feature 4: Temp Scaling
        else:
            self.current_window_size = min(self.max_window_size, self.current_window_size + 5)
            recal_freq = self.recalibrate_stable
            self.dynamic_temp = max(1.0, self.dynamic_temp - 0.05)

        # Get the final calibrated prediction for THIS sample
        pred_conf = self.calibrate(confidence)
        self.history.append((confidence, pred_conf, outcome, weight))
        
        while len(self.history) > self.current_window_size:
            self.history.pop(0)
            
        self.updates_since_cal += 1
        if self.updates_since_cal >= recal_freq and len(self.history) > 10:
            self._fit()
            self.updates_since_cal = 0
            
    def _temp_scale(self, raw_confidence: float) -> float:
        eps = 1e-7
        conf_clip = np.clip(raw_confidence, eps, 1-eps)
        logit = np.log(conf_clip / (1 - conf_clip))
        scaled_logit = logit / self.dynamic_temp
        return float(1.0 / (1.0 + np.exp(-scaled_logit)))

    def _fit(self):
        conf_arr = np.array([self._temp_scale(x[0]) for x in self.history])
        out_arr = np.array([x[2] for x in self.history])
        weights = np.array([x[3] * (self.decay_factor ** (len(self.history) - i - 1)) 
                            for i, x in enumerate(self.history)])
        self.model = IsotonicRegression(out_of_bounds='clip')
        conf_arr_noise = conf_arr + np.random.normal(0, 1e-6, len(conf_arr))
        self.model.fit(conf_arr_noise, out_arr, sample_weight=weights)
        
    def calibrate(self, raw_confidence: float) -> float:
        temp_scaled_conf = self._temp_scale(raw_confidence)
        
        if self.model is None or len(self.history) < 10:
            iso_conf = temp_scaled_conf
        else:
            iso_conf = float(self.model.predict([temp_scaled_conf])[0])
            
        # Feature 3: Confidence Decay
        if self.state in [CalibrationState.DRIFTING, CalibrationState.COLLAPSED]:
            iso_conf *= np.exp(-self.alpha_decay * self.drift_score)
            
        return float(np.clip(iso_conf, 0.0, 1.0))
        
    def ece(self, n_bins: int = 10) -> float:
        if len(self.history) < 10:
            return 0.0
        
        # ECE must evaluate the historical PREDICTED confidence vs actual outcome
        cal_conf = np.array([x[1] for x in self.history])
        out_arr = np.array([float(x[2]) for x in self.history])
        
        # Apply exponential weighting to ECE matching the fit decay
        weights = np.array([self.decay_factor ** (len(self.history) - i - 1) for i in range(len(self.history))])
        weights = weights / np.sum(weights)
            
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (cal_conf >= lower) & (cal_conf <= upper)
            prop = np.sum(weights[in_bin])
            if prop > 0:
                acc = np.sum(out_arr[in_bin] * weights[in_bin]) / np.sum(weights[in_bin])
                avg_conf = np.sum(cal_conf[in_bin] * weights[in_bin]) / np.sum(weights[in_bin])
                ece_val += np.abs(avg_conf - acc) * prop
        return float(ece_val)
        
    @property
    def is_surgery_frozen(self) -> bool:
        # Feature 5: Emergency Brake
        return self.state == CalibrationState.COLLAPSED
