"""
benchmarks.evaluators
=====================
Statistically rigorous evaluations for causal structure recovery.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .generators import BenchmarkGraph

@dataclass
class BenchmarkReport:
    detection_delay: float
    shd: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    runtime_ms: float

class BenchmarkEvaluator:
    """
    Computes objective metrics comparing predicted graphs against ground truth.
    """
    
    @staticmethod
    def _get_binary_adj(graph: BenchmarkGraph) -> np.ndarray:
        return (np.abs(graph.adj_matrix) > 1e-3).astype(int)

    def evaluate_structure(self, pred_adj: np.ndarray, true_graph: BenchmarkGraph) -> Dict[str, float]:
        """Compute SHD, Precision, Recall, F1"""
        true_adj = self._get_binary_adj(true_graph)
        pred_adj = (np.abs(pred_adj) > 1e-3).astype(int)
        
        diff = pred_adj - true_adj
        shd = float(np.sum(np.abs(diff)))
        
        true_pos = np.sum((pred_adj == 1) & (true_adj == 1))
        false_pos = np.sum((pred_adj == 1) & (true_adj == 0))
        false_neg = np.sum((pred_adj == 0) & (true_adj == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "shd": shd,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    def evaluate_online(self, 
                        detected_cycle: Optional[int], 
                        ground_truth_cycle: int,
                        alarms_fired: List[int],
                        horizon: int) -> Dict[str, float]:
        """Compute detection delay and false alarm rate"""
        if detected_cycle is None or detected_cycle < ground_truth_cycle:
            delay = float('inf')
        else:
            delay = float(detected_cycle - ground_truth_cycle)
            
        # Alarms outside the window [gt_cycle, gt_cycle + max_delay] are false alarms
        max_delay = 50
        false_alarms = sum(1 for c in alarms_fired if c < ground_truth_cycle or c > ground_truth_cycle + max_delay)
        far = false_alarms / len(alarms_fired) if alarms_fired else 0.0
        
        return {
            "detection_delay": delay,
            "false_alarm_rate": float(far)
        }
