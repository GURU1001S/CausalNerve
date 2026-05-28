import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

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

class FleetEpidemiologyGraph:
    """Global graph of engines, motifs, and interventions."""
    def __init__(self):
        # motif_id -> list of occurrences
        self.motif_occurrences: Dict[str, List[MotifOccurrence]] = defaultdict(list)
        # motif_id -> list of interventions
        self.interventions: Dict[str, List[InterventionEvent]] = defaultdict(list)
        # engine_id -> set of active motifs
        self.engine_state: Dict[str, Set[str]] = defaultdict(set)
        
    def log_occurrence(self, engine_id: str, motif_fingerprint: str, cycle: int, severity: float):
        occ = MotifOccurrence(engine_id, motif_fingerprint, cycle, severity)
        self.motif_occurrences[motif_fingerprint].append(occ)
        self.engine_state[engine_id].add(motif_fingerprint)

    def log_intervention(self, engine_id: str, motif_fingerprint: str, edge: tuple, success: bool, cycle: int):
        evt = InterventionEvent(engine_id, motif_fingerprint, edge, success, cycle)
        self.interventions[motif_fingerprint].append(evt)
        if success and motif_fingerprint in self.engine_state[engine_id]:
            self.engine_state[engine_id].remove(motif_fingerprint)

class MotifPropagationTracker:
    @staticmethod
    def get_most_contagious(graph: FleetEpidemiologyGraph) -> Optional[str]:
        if not graph.motif_occurrences:
            return None
        # Metric: number of distinct engines infected
        return max(graph.motif_occurrences.keys(), 
                   key=lambda k: len(set(o.engine_id for o in graph.motif_occurrences[k])))

    @staticmethod
    def compute_prevalence(graph: FleetEpidemiologyGraph) -> Dict[str, float]:
        total_engines = max(1, len(graph.engine_state))
        prevalence = {}
        for motif, occs in graph.motif_occurrences.items():
            infected = len(set(o.engine_id for o in occs if motif in graph.engine_state[o.engine_id]))
            prevalence[motif] = infected / total_engines
        return prevalence

class TransferLearningLayer:
    @staticmethod
    def get_recommended_interventions(graph: FleetEpidemiologyGraph, engine_id: str) -> List[dict]:
        recommendations = []
        active_motifs = graph.engine_state.get(engine_id, set())
        for motif in active_motifs:
            # Find successful interventions for this motif across the fleet
            successes = [i for i in graph.interventions.get(motif, []) if i.was_successful]
            if successes:
                # Naive: pick the most common successful edge
                edge_counts = defaultdict(int)
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

class FleetRiskForecaster:
    @staticmethod
    def compute_fleet_stability_index(graph: FleetEpidemiologyGraph) -> float:
        if not graph.engine_state:
            return 1.0
        # Inverse to the number of active motifs per engine
        total_motifs_active = sum(len(motifs) for motifs in graph.engine_state.values())
        avg_motifs = total_motifs_active / len(graph.engine_state)
        # Normalize arbitrarily for index 0 to 1
        return max(0.0, 1.0 - (avg_motifs / 5.0))

class EpidemiologyEngine:
    def __init__(self):
        self.graph = FleetEpidemiologyGraph()
        
    def process_live_telemetry(self, engine_id: str, cycle: int, active_motifs: List[dict]):
        for m in active_motifs:
            fid = m.get("motif_fingerprint", "unknown")
            self.graph.log_occurrence(engine_id, fid, cycle, severity=m.get("similarity", 0.5))

    def get_dashboard_metrics(self, current_engine_id: str) -> dict:
        contagious = MotifPropagationTracker.get_most_contagious(self.graph)
        prevalence = MotifPropagationTracker.compute_prevalence(self.graph)
        stability = FleetRiskForecaster.compute_fleet_stability_index(self.graph)
        recommendations = TransferLearningLayer.get_recommended_interventions(self.graph, current_engine_id)
        
        # Format for UI
        clusters = [{"motif": k, "infected": len(set(o.engine_id for o in v))} for k, v in self.graph.motif_occurrences.items()]
        clusters.sort(key=lambda x: x["infected"], reverse=True)
        
        return {
            "most_contagious": contagious,
            "fleet_stability_index": stability,
            "motif_clusters": clusters[:5],
            "prevalence_map": prevalence,
            "transfer_recommendations": recommendations,
            "total_tracked_engines": len(self.graph.engine_state)
        }
