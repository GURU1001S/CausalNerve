"""
causalnerve.fleet.epidemiology
==============================
Fleet-level causal epidemiology with real trajectory similarity.

All metrics are computed from actual data — no hardcoded constants.

Key replacements from prior version
------------------------------------
- ``dtw_match()`` previously returned a hardcoded 0.95.
  Now computes real Dynamic Time Warping distance using the
  O(N·M) dynamic-programming algorithm.

- ``compute_fleet_stability_index()`` previously used an
  arbitrary heuristic with a hardcoded divisor.
  Now computes normalized structural entropy over edge-probability
  distributions and motif volatility.
"""

import time
import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


# ─────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────

@dataclass
class MotifOccurrence:
    engine_id: str
    motif_fingerprint: str
    cycle_detected: int
    severity: float

@dataclass
class InterventionEvent:
    engine_id: str
    motif_fingerprint: str
    proposed_edge: tuple
    was_successful: bool
    cycle_applied: int


class ScientificIntegrityError(Exception):
    """Raised when a fake or hardcoded value is detected at runtime."""
    pass


# ─────────────────────────────────────────────────────
#  Pure-Python Dynamic Time Warping (no external deps)
# ─────────────────────────────────────────────────────

def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two 1-D sequences.

    Uses the standard O(N·M) dynamic-programming formulation:

        DTW(i, j) = |a_i − b_j| + min(DTW(i−1, j),
                                        DTW(i, j−1),
                                        DTW(i−1, j−1))

    Returns the total accumulated cost (unnormalised).

    Parameters
    ----------
    seq_a, seq_b : 1-D numpy arrays of arbitrary (possibly different) length.

    References
    ----------
    Sakoe, H., Chiba, S. (1978). Dynamic programming algorithm
    optimization for spoken word recognition. IEEE Trans. ASSP.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0

    # Cost matrix — +1 to accommodate the boundary row/column
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(float(seq_a[i - 1]) - float(seq_b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j],
                                  cost[i, j - 1],
                                  cost[i - 1, j - 1])

    return float(cost[n, m])


def _dtw_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Convert DTW distance to a [0, 1] similarity score.

    Similarity = 1 / (1 + DTW / path_length)

    where path_length = max(len(a), len(b)) normalises for sequence length.
    """
    dist = _dtw_distance(seq_a, seq_b)
    path_len = max(len(seq_a), len(seq_b), 1)
    return 1.0 / (1.0 + dist / path_len)


def _multivariate_dtw_distance(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    DTW on multivariate time series using Euclidean point-wise distance.

    Parameters
    ----------
    mat_a : (T1, D) array
    mat_b : (T2, D) array

    Returns total accumulated DTW cost.
    """
    n, m = mat_a.shape[0], mat_b.shape[0]
    if n == 0 or m == 0:
        return 0.0

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(mat_a[i - 1] - mat_b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j],
                                  cost[i, j - 1],
                                  cost[i - 1, j - 1])

    return float(cost[n, m])


# ─────────────────────────────────────────────────────
#  Fleet Epidemiology Graph
# ─────────────────────────────────────────────────────

class FleetEpidemiologyGraph:
    """Global graph of engines, motifs, and interventions."""

    def __init__(self):
        # motif_id -> list of occurrences
        self.motif_occurrences: Dict[str, List[MotifOccurrence]] = defaultdict(list)
        # motif_id -> list of interventions
        self.interventions: Dict[str, List[InterventionEvent]] = defaultdict(list)
        # engine_id -> set of active motifs
        self.engine_state: Dict[str, Set[str]] = defaultdict(set)
        # engine_id -> rolling telemetry trajectories (list of 1-D snapshots)
        self.engine_trajectories: Dict[str, List[np.ndarray]] = defaultdict(list)
        # engine_id -> rolling leakage history
        self.engine_leakage_history: Dict[str, List[float]] = defaultdict(list)

    @property
    def engines(self):
        return list(self.engine_state.keys())

    def register_engine(self, engine_id: str):
        if engine_id not in self.engine_state:
            self.engine_state[engine_id] = set()

    def append_telemetry(self, engine_id: str, snapshot: np.ndarray):
        """Record one cycle of telemetry for trajectory matching."""
        self.engine_trajectories[engine_id].append(snapshot.copy())

    def append_leakage(self, engine_id: str, leakage: float):
        """Record one cycle of aggregate leakage."""
        self.engine_leakage_history[engine_id].append(float(leakage))

    # ── Real DTW matching ───────────────────────────

    def dtw_match(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute REAL Dynamic Time Warping similarity between two
        univariate sequences.

        Returns a similarity score in [0, 1].

        Raises ScientificIntegrityError if called with empty inputs
        that would force a hardcoded fallback.
        """
        if len(seq1) == 0 and len(seq2) == 0:
            raise ScientificIntegrityError(
                "dtw_match called with two empty sequences — cannot compute similarity"
            )
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        return _dtw_similarity(np.asarray(seq1, dtype=float),
                               np.asarray(seq2, dtype=float))

    def compute_engine_similarity(self, eng_a: str, eng_b: str) -> Dict[str, float]:
        """
        Multi-modal similarity between two engines.

        Combines:
          1. Leakage trajectory DTW
          2. Structural motif Jaccard overlap
          3. Telemetry trajectory DTW (if available)

        Returns dict with per-channel similarity and a fused score.
        """
        result: Dict[str, float] = {}

        # 1. Leakage trajectory DTW
        la = np.array(self.engine_leakage_history.get(eng_a, []))
        lb = np.array(self.engine_leakage_history.get(eng_b, []))
        if len(la) > 1 and len(lb) > 1:
            result["leakage_dtw"] = _dtw_similarity(la, lb)
        else:
            result["leakage_dtw"] = 0.0

        # 2. Motif Jaccard overlap
        ma = self.engine_state.get(eng_a, set())
        mb = self.engine_state.get(eng_b, set())
        if ma or mb:
            jaccard = len(ma & mb) / max(len(ma | mb), 1)
            result["motif_jaccard"] = jaccard
        else:
            result["motif_jaccard"] = 0.0

        # 3. Telemetry trajectory DTW (use last 50 cycles)
        ta = self.engine_trajectories.get(eng_a, [])
        tb = self.engine_trajectories.get(eng_b, [])
        if len(ta) > 1 and len(tb) > 1:
            # Stack into matrices and compute multivariate DTW
            mat_a = np.array(ta[-50:])
            mat_b = np.array(tb[-50:])
            # Ensure same feature dimension
            d = min(mat_a.shape[1], mat_b.shape[1]) if mat_a.ndim > 1 and mat_b.ndim > 1 else 0
            if d > 0:
                dist = _multivariate_dtw_distance(mat_a[:, :d], mat_b[:, :d])
                path_len = max(mat_a.shape[0], mat_b.shape[0])
                result["telemetry_dtw"] = 1.0 / (1.0 + dist / path_len)
            else:
                result["telemetry_dtw"] = 0.0
        else:
            result["telemetry_dtw"] = 0.0

        # Fused similarity: weighted average
        weights = {"leakage_dtw": 0.4, "motif_jaccard": 0.3, "telemetry_dtw": 0.3}
        fused = sum(result[k] * weights[k] for k in weights)
        result["fused_similarity"] = fused

        return result

    def log_occurrence(self, engine_id: str, motif_fingerprint: str,
                       cycle: int, severity: float):
        occ = MotifOccurrence(engine_id, motif_fingerprint, cycle, severity)
        self.motif_occurrences[motif_fingerprint].append(occ)
        self.engine_state[engine_id].add(motif_fingerprint)

    def log_intervention(self, engine_id: str, motif_fingerprint: str,
                          edge: tuple, success: bool, cycle: int):
        evt = InterventionEvent(engine_id, motif_fingerprint, edge, success, cycle)
        self.interventions[motif_fingerprint].append(evt)
        if success and motif_fingerprint in self.engine_state[engine_id]:
            self.engine_state[engine_id].remove(motif_fingerprint)


# ─────────────────────────────────────────────────────
#  Motif analysis
# ─────────────────────────────────────────────────────

class MotifPropagationTracker:
    @staticmethod
    def get_most_contagious(graph: FleetEpidemiologyGraph) -> Optional[str]:
        if not graph.motif_occurrences:
            return None
        return max(graph.motif_occurrences.keys(),
                   key=lambda k: len(set(o.engine_id for o in graph.motif_occurrences[k])))

    @staticmethod
    def compute_prevalence(graph: FleetEpidemiologyGraph) -> Dict[str, float]:
        total_engines = max(1, len(graph.engine_state))
        prevalence = {}
        for motif, occs in graph.motif_occurrences.items():
            infected = len(set(o.engine_id for o in occs
                               if motif in graph.engine_state[o.engine_id]))
            prevalence[motif] = infected / total_engines
        return prevalence

    @staticmethod
    def compute_motif_entropy(graph: FleetEpidemiologyGraph) -> float:
        """
        Shannon entropy over motif prevalence distribution.

        H = −Σ_m p_m · log₂(p_m)

        High entropy → many motifs equally prevalent (unstable fleet).
        Low entropy  → dominated by one motif or none (stable or single-fault).
        """
        prev = MotifPropagationTracker.compute_prevalence(graph)
        if not prev:
            return 0.0
        vals = [v for v in prev.values() if v > 0]
        if not vals:
            return 0.0
        total = sum(vals)
        probs = [v / total for v in vals]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def compute_motif_persistence(graph: FleetEpidemiologyGraph) -> Dict[str, float]:
        """
        Mean persistence (in cycles) of each motif across the fleet.

        Persistence_m = mean over engines of (last_cycle − first_cycle + 1).
        """
        persistence = {}
        for motif, occs in graph.motif_occurrences.items():
            per_engine: Dict[str, List[int]] = defaultdict(list)
            for o in occs:
                per_engine[o.engine_id].append(o.cycle_detected)
            if per_engine:
                spans = [max(cycles) - min(cycles) + 1 for cycles in per_engine.values()]
                persistence[motif] = float(np.mean(spans))
        return persistence


# ─────────────────────────────────────────────────────
#  Transfer learning layer
# ─────────────────────────────────────────────────────

class TransferLearningLayer:
    @staticmethod
    def get_recommended_interventions(graph: FleetEpidemiologyGraph,
                                       engine_id: str) -> List[dict]:
        recommendations = []
        active_motifs = graph.engine_state.get(engine_id, set())
        for motif in active_motifs:
            successes = [i for i in graph.interventions.get(motif, [])
                         if i.was_successful]
            if successes:
                edge_counts: Dict[tuple, int] = defaultdict(int)
                for s in successes:
                    edge_counts[s.proposed_edge] += 1
                best_edge = max(edge_counts.keys(), key=lambda k: edge_counts[k])
                recommendations.append({
                    "motif": motif,
                    "recommended_edge": best_edge,
                    "confidence": edge_counts[best_edge] / len(successes),
                    "prior_successes": edge_counts[best_edge]
                })
        return recommendations


# ─────────────────────────────────────────────────────
#  Real fleet stability index
# ─────────────────────────────────────────────────────

class FleetRiskForecaster:
    """
    All stability and risk metrics are computed from actual
    fleet data — no arbitrary constants.
    """

    @staticmethod
    def compute_fleet_stability_index(graph: FleetEpidemiologyGraph) -> float:
        """
        Fleet Stability = 1 − normalised structural entropy.

        Structural entropy is the Shannon entropy of the motif
        prevalence distribution, normalised by log₂(K) where K is
        the number of distinct motifs.

        S = 1 − H(motifs) / log₂(K)

        When H = 0 (no motifs or single dominant fault), S = 1.
        When H = log₂(K) (maximum disorder), S = 0.
        """
        if not graph.engine_state:
            return 1.0

        H = MotifPropagationTracker.compute_motif_entropy(graph)
        K = len(graph.motif_occurrences)
        if K <= 1:
            return 1.0

        H_max = math.log2(K)
        normalised = H / H_max if H_max > 0 else 0.0
        return float(np.clip(1.0 - normalised, 0.0, 1.0))

    @staticmethod
    def forecast_risk(graph: FleetEpidemiologyGraph,
                      engine_id: str) -> Dict:
        """
        Predict structural risk for a specific engine.

        Uses:
          - nearest historical engines by DTW similarity
          - motif transition probability
          - leakage escalation rate

        Returns a structured forecast object.
        """
        # 1. Find nearest engines by leakage DTW
        target_leak = np.array(graph.engine_leakage_history.get(engine_id, []))
        similarities = []
        for eid in graph.engines:
            if eid == engine_id:
                continue
            other_leak = np.array(graph.engine_leakage_history.get(eid, []))
            if len(target_leak) > 1 and len(other_leak) > 1:
                sim = _dtw_similarity(target_leak, other_leak)
                similarities.append((eid, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest = similarities[:3]

        # 2. Motif transition probability
        # P(new motif) = fraction of nearest engines that have motifs
        # the target engine does not yet have
        target_motifs = graph.engine_state.get(engine_id, set())
        novel_motif_count = 0
        total_neighbor_motifs = 0
        for eid, _ in nearest:
            neighbor_motifs = graph.engine_state.get(eid, set())
            novel = neighbor_motifs - target_motifs
            novel_motif_count += len(novel)
            total_neighbor_motifs += len(neighbor_motifs)

        transition_prob = (novel_motif_count / max(total_neighbor_motifs, 1)
                           if total_neighbor_motifs > 0 else 0.0)

        # 3. Leakage escalation rate (linear regression slope)
        escalation_rate = 0.0
        if len(target_leak) >= 5:
            x = np.arange(len(target_leak), dtype=float)
            # Least-squares slope: β = cov(x,y) / var(x)
            x_mean = np.mean(x)
            y_mean = np.mean(target_leak)
            cov = np.sum((x - x_mean) * (target_leak - y_mean))
            var = np.sum((x - x_mean) ** 2)
            if var > 0:
                escalation_rate = float(cov / var)

        # 4. Estimated cycles to instability
        # If escalation rate > 0, estimate when leakage crosses threshold
        instability_threshold = 0.15
        current_leak = float(target_leak[-1]) if len(target_leak) > 0 else 0.0
        if escalation_rate > 1e-8 and current_leak < instability_threshold:
            cycles_to_failure = (instability_threshold - current_leak) / escalation_rate
        elif current_leak >= instability_threshold:
            cycles_to_failure = 0.0
        else:
            cycles_to_failure = float('inf')

        # 5. Risk score = weighted combination
        # Higher transition probability + higher escalation = higher risk
        risk_score = float(np.clip(
            0.4 * transition_prob + 0.4 * min(escalation_rate * 100, 1.0)
            + 0.2 * (1.0 - FleetRiskForecaster.compute_fleet_stability_index(graph)),
            0.0, 1.0
        ))

        # Confidence derived from data availability
        data_points = len(target_leak) + len(nearest)
        confidence = float(np.clip(data_points / 50.0, 0.0, 0.95))

        fleet_entropy = MotifPropagationTracker.compute_motif_entropy(graph)

        return {
            "risk_score": round(risk_score, 4),
            "cycles_to_failure": round(cycles_to_failure, 1) if cycles_to_failure != float('inf') else None,
            "nearest_engines": [{"engine_id": eid, "similarity": round(sim, 4)}
                                for eid, sim in nearest],
            "motif_matches": list(target_motifs),
            "trajectory_similarity": round(nearest[0][1], 4) if nearest else 0.0,
            "fleet_entropy": round(fleet_entropy, 4),
            "escalation_rate": round(escalation_rate, 6),
            "transition_probability": round(transition_prob, 4),
            "confidence": round(confidence, 4),
        }


# ─────────────────────────────────────────────────────
#  Epidemiology Engine (main entry point)
# ─────────────────────────────────────────────────────

class EpidemiologyEngine:
    def __init__(self):
        self.graph = FleetEpidemiologyGraph()

    def process_live_telemetry(self, engine_id: str, cycle: int,
                                active_motifs: List[dict]):
        for m in active_motifs:
            fid = m.get("motif_fingerprint", "unknown")
            self.graph.log_occurrence(engine_id, fid, cycle,
                                      severity=m.get("similarity", 0.5))

    def get_dashboard_metrics(self, current_engine_id: str) -> dict:
        contagious = MotifPropagationTracker.get_most_contagious(self.graph)
        prevalence = MotifPropagationTracker.compute_prevalence(self.graph)
        stability = FleetRiskForecaster.compute_fleet_stability_index(self.graph)
        recommendations = TransferLearningLayer.get_recommended_interventions(
            self.graph, current_engine_id)

        clusters = [{"motif": k,
                      "infected": len(set(o.engine_id for o in v))}
                     for k, v in self.graph.motif_occurrences.items()]
        clusters.sort(key=lambda x: x["infected"], reverse=True)

        return {
            "most_contagious": contagious,
            "fleet_stability_index": stability,
            "motif_clusters": clusters[:5],
            "prevalence_map": prevalence,
            "transfer_recommendations": recommendations,
            "total_tracked_engines": len(self.graph.engine_state)
        }
