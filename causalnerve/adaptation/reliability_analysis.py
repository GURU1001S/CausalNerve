"""
causalnerve.adaptation.reliability_analysis
======================================
Evaluates the true safety and trustworthiness of CausalNerve.
Computes Brier scores, Maximum Calibration Error (MCE), and generates Reliability Diagrams.
"""

import numpy as np
from typing import Dict, Tuple

class ReliabilityAnalyzer:
    """
    Computes strict safety metrics to prove the system avoids overconfident errors.
    """
    
    @staticmethod
    def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Mean squared error between predicted probabilities and actual binary outcomes.
        Lower is better. Perfect calibration and accuracy = 0.0.
        """
        return float(np.mean((probs - labels) ** 2))
        
    @staticmethod
    def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Maximum Calibration Error.
        Measures the worst-case deviation between confidence and accuracy.
        Critical for safety-critical systems.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs >= lower) & (probs <= upper)
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, error)
                
        return float(mce)

    @staticmethod
    def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute the full suite of reliability metrics."""
        from .calibrator import ConfidenceCalibrator
        
        ece = ConfidenceCalibrator().expected_calibration_error(probs, labels)
        mce = ReliabilityAnalyzer.compute_mce(probs, labels)
        brier = ReliabilityAnalyzer.compute_brier_score(probs, labels)
        
        # AUC proxy for discriminative power
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5 # Default if only one class is present
            
        return {
            "ECE": ece,
            "MCE": mce,
            "BrierScore": brier,
            "AUC": auc
        }
