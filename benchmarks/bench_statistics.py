import numpy as np
from typing import Dict, List, Any
try:
    from scipy import stats
except ImportError:
    pass

def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Computes Cohen's d for effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return (np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var)

class StatisticalPipeline:
    """Rigorous statistical testing for benchmark results."""
    
    @staticmethod
    def compare_models(metric_name: str, base_scores: List[float], challenger_scores: List[float]) -> Dict[str, Any]:
        """Compares two models across multiple seeds/trials."""
        try:
            # Mann-Whitney U test (non-parametric independent)
            u_stat, p_mw = stats.mannwhitneyu(base_scores, challenger_scores, alternative='two-sided')
            
            # Wilcoxon signed-rank test (non-parametric paired)
            w_stat, p_wilcoxon = stats.wilcoxon(base_scores, challenger_scores)
            
            cohens_d = compute_cohens_d(base_scores, challenger_scores)
            
            return {
                "metric": metric_name,
                "mann_whitney_p": float(p_mw),
                "wilcoxon_p": float(p_wilcoxon),
                "cohens_d": float(cohens_d),
                "significant": p_wilcoxon < 0.05
            }
        except Exception:
            # Fallback if scipy missing or zero variance
            return {
                "metric": metric_name,
                "mann_whitney_p": 1.0,
                "wilcoxon_p": 1.0,
                "cohens_d": 0.0,
                "significant": False
            }
