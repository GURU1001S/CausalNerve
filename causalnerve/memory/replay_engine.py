import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class GraphSnapshot:
    cycle: int
    adjacency: Any
    leakage: float
    v_energy: float

@dataclass
class RevisionRecord:
    cycle: int
    edit_type: str
    edge: Tuple[int, int]
    confidence: float
    v_before: float
    v_after: float
    rationale: str = ""
    accepted: bool = True

@dataclass
class GraphDiff:
    edges_added: List[Tuple[int, int]]
    edges_removed: List[Tuple[int, int]]
    weight_changes: Dict[Tuple[int, int], float]

class StructuralReplayEngine:
    def __init__(self, snapshot_interval: int = 10):
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[GraphSnapshot] = []
        self.revision_events: List[RevisionRecord] = []

    def _safe_store(self, tensor: Any) -> Any:
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy().copy()
        return np.asarray(tensor).copy()

    def record_snapshot(self, cycle: int, adjacency: Any, leakage: float, v_energy: float):
        adj_safe = self._safe_store(adjacency)
        snap = GraphSnapshot(
            cycle=cycle,
            adjacency=adj_safe,
            leakage=float(leakage),
            v_energy=float(v_energy)
        )
        self.snapshots.append(snap)

    def record_revision(self, cycle: int, edit_type: str, edge: Tuple[int, int], confidence: float, v_before: float, v_after: float, rationale: str = "", accepted: bool = True):
        rev = RevisionRecord(
            cycle=cycle,
            edit_type=edit_type,
            edge=edge,
            confidence=float(confidence),
            v_before=float(v_before),
            v_after=float(v_after),
            rationale=rationale,
            accepted=accepted
        )
        self.revision_events.append(rev)

    def get_snapshot(self, cycle: int) -> Optional[GraphSnapshot]:
        best_snap = None
        for snap in self.snapshots:
            if snap.cycle <= cycle:
                if best_snap is None or snap.cycle > best_snap.cycle:
                    best_snap = snap
        return best_snap
        
    def get_graph_diff(self, cycle_a: int, cycle_b: int) -> GraphDiff:
        added = []
        removed = []
        for rev in self.revision_events:
            if cycle_a <= rev.cycle <= cycle_b and rev.accepted:
                if rev.edit_type == "add":
                    added.append(rev.edge)
                elif rev.edit_type == "remove":
                    removed.append(rev.edge)
        return GraphDiff(edges_added=added, edges_removed=removed, weight_changes={})
        
    def get_revision_events_in_range(self, start_cycle: int, end_cycle: int):
        return [r for r in self.revision_events if start_cycle <= r.cycle <= end_cycle]

    def summary(self) -> Dict[str, Any]:
        return {
            "total_snapshots": len(self.snapshots),
            "total_revisions": len(self.revision_events),
            "latest_cycle": self.snapshots[-1].cycle if self.snapshots else 0
        }
