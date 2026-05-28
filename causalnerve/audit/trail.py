from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time

@dataclass
class EvidencePacket:
    """Stores all empirical evidence justifying a causal decision."""
    leakage_before: float
    leakage_after: float
    confidence_score: float
    counterfactual_divergence: float
    motif_similarity: float
    physical_constraints_triggered: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "leakage_before": self.leakage_before,
            "leakage_after": self.leakage_after,
            "confidence_score": self.confidence_score,
            "counterfactual_divergence": self.counterfactual_divergence,
            "motif_similarity": self.motif_similarity,
            "physical_constraints_triggered": self.physical_constraints_triggered
        }

@dataclass
class RootCauseNode:
    """A node in the causal intervention chain."""
    event_id: str
    action: str # ADD, REJECT, ROLLBACK
    edge: tuple
    reasoning: str
    evidence: EvidencePacket
    dependencies: List[str] = field(default_factory=list) # List of event_ids

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "action": self.action,
            "edge": self.edge,
            "reasoning": self.reasoning,
            "evidence": self.evidence.to_dict(),
            "dependencies": self.dependencies
        }

class RootCauseTree:
    """Visualizes and tracks causal chains and intervention dependencies."""
    def __init__(self):
        self.nodes: Dict[str, RootCauseNode] = {}
        
    def add_node(self, node: RootCauseNode):
        self.nodes[node.event_id] = node
        
    def get_chain(self, leaf_event_id: str) -> List[RootCauseNode]:
        """Walks up the dependency tree to find root causes."""
        chain = []
        current = leaf_event_id
        while current in self.nodes:
            node = self.nodes[current]
            chain.append(node)
            if not node.dependencies:
                break
            current = node.dependencies[0] # Simplification for single-parent paths
        return chain

class AuditTrail:
    """Central repository for all scientific causal audits."""
    def __init__(self):
        self.tree = RootCauseTree()
        self.history: List[RootCauseNode] = []
        
    def log_surgery(self, event_id: str, action: str, edge: tuple, reasoning: str, evidence: EvidencePacket, dependencies: List[str] = None):
        """Logs a highly explainable graph surgery event."""
        node = RootCauseNode(
            event_id=event_id,
            action=action,
            edge=edge,
            reasoning=reasoning,
            evidence=evidence,
            dependencies=dependencies or []
        )
        self.tree.add_node(node)
        self.history.append(node)
        return node
        
    def get_full_audit(self) -> List[Dict[str, Any]]:
        return [n.to_dict() for n in self.history]
