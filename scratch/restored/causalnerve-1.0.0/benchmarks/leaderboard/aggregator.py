import numpy as np
from typing import Dict, List, Any
try:
    from scipy import stats
except ImportError:
    stats = None

class ResultsAggregator:
    """Calculates means, stds, confidence intervals and significance for leaderboard."""
    
    @staticmethod
    def compute_stats(metrics_list: List[float]) -> Dict[str, float]:
        """Compute robust statistics including 95% CI."""
        if not metrics_list:
            return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            
        data = np.array(metrics_list)
        mean = np.mean(data)
        std = np.std(data, ddof=1) if len(data) > 1 else 0.0
        
        # 95% CI using standard error (assuming normal dist or large N)
        n = len(data)
        if n > 1 and stats is not None:
            se = std / np.sqrt(n)
            ci = stats.t.ppf(0.975, n-1) * se
        else:
            ci = 0.0 # Fallback
            
        return {
            "mean": float(mean),
            "std": float(std),
            "ci_lower": float(mean - ci),
            "ci_upper": float(mean + ci)
        }

    @staticmethod
    def aggregate(raw_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Input: { "VARLiNGAM": { "SHD": [2, 3, 1], "Runtime": [0.5, 0.6, 0.4] } }
        Output: Aggregated metrics ready for leaderboard visualization.
        """
        aggregated = {}
        for model_name, metrics in raw_results.items():
            aggregated[model_name] = {}
            for metric_name, values in metrics.items():
                aggregated[model_name][metric_name] = ResultsAggregator.compute_stats(values)
        return aggregated
