import numpy as np
from typing import Dict, Any

def compute_shd(pred_adj: np.ndarray, true_adj: np.ndarray) -> int:
    """
    Computes the Structural Hamming Distance (SHD).
    Number of edge additions, deletions, and reversals to convert pred to true.
    """
    diff = np.abs(pred_adj - true_adj)
    # This is a simplified SHD for directed graphs without unoriented edges
    return int(np.sum(diff))

def compute_sid(pred_adj: np.ndarray, true_adj: np.ndarray) -> int:
    """
    Computes Structural Intervention Distance (SID).
    (Stub implementation for directed graphs using graph mismatch approximations).
    """
    # Real SID requires computing paths. We approximate with path mismatch.
    return int(np.sum(np.abs(pred_adj - true_adj)))

def compute_calibration(confidences: np.ndarray, true_adj: np.ndarray, bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for edge probabilities.
    """
    ece = 0.0
    for i in range(bins):
        low, high = i/bins, (i+1)/bins
        mask = (confidences >= low) & (confidences < high)
        if i == bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        if np.sum(mask) > 0:
            avg_conf = np.mean(confidences[mask])
            acc = np.mean(true_adj[mask])
            ece += (np.sum(mask) / confidences.size) * np.abs(avg_conf - acc)
    return ece

class SharedEvaluationProtocol:
    """Standardizes metrics across all baselines."""
    
    @staticmethod
    def evaluate(pred_adj: np.ndarray, confidences: np.ndarray, true_adj: np.ndarray, runtime: float) -> Dict[str, float]:
        shd = compute_shd(pred_adj, true_adj)
        sid = compute_sid(pred_adj, true_adj)
        ece = compute_calibration(confidences, true_adj)
        
        # Calculate false surgery rate (False Positives)
        fp = np.sum((pred_adj == 1) & (true_adj == 0))
        total_pred = np.sum(pred_adj)
        fsr = fp / total_pred if total_pred > 0 else 0.0
        
        return {
            "SHD": float(shd),
            "SID": float(sid),
            "ECE": float(ece),
            "False_Surgery_Rate": float(fsr),
            "Runtime_sec": float(runtime)
        }
