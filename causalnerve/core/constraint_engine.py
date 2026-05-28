import json
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

class ConstraintType(Enum):
    TEMPORAL_PRECEDENCE = "Temporal Precedence"
    THERMODYNAMIC_DIRECTION = "Thermodynamic Directionality"
    CONSERVATION_VIOLATION = "Conservation Violation"
    SENSOR_IMPLAUSIBILITY = "Sensor Implausibility"
    DOMAIN_FORBIDDEN = "Domain Forbidden Registry"

@dataclass
class EvaluationResult:
    is_valid: bool
    score: float
    confidence: float
    violations: List[str]
    explanation: str

@dataclass
class PhysicsNode:
    id: int
    name: str
    component_type: str  # e.g., 'compressor', 'turbine', 'sensor', 'combustor'
    flow_stage: int      # 1 to N (upstream to downstream)
    temperature_tier: int # 1 to N (cool to hot)

class PhysicalConstraintEngine:
    """
    A modular, domain-agnostic constraint engine for causal intelligence.
    Currently configured for NASA C-MAPSS FD004 thermodynamic topologies.
    """
    def __init__(self):
        self.nodes: Dict[int, PhysicsNode] = {}
        self.forbidden_edges: Set[Tuple[int, int]] = set()
        
        # Dashboard reporting state
        self.total_evaluations: int = 0
        self.total_violations: int = 0
        self.rejected_edges: List[Dict[str, Any]] = []
        
        self._load_cmapss_domain()

    def _load_cmapss_domain(self):
        """Loads FD004 specific physical topology."""
        cmapss_nodes = {
            0:  PhysicsNode(0, "Fan", "compressor", 1, 1),
            1:  PhysicsNode(1, "LPC", "compressor", 2, 2),
            2:  PhysicsNode(2, "HPC", "compressor", 3, 3),
            3:  PhysicsNode(3, "Combustor", "combustor", 4, 5),
            4:  PhysicsNode(4, "HPT", "turbine", 5, 4),
            5:  PhysicsNode(5, "LPT", "turbine", 6, 3),
            6:  PhysicsNode(6, "H.Spool", "mechanical", 0, 0),
            7:  PhysicsNode(7, "L.Spool", "mechanical", 0, 0),
            8:  PhysicsNode(8, "P.Bank", "mechanical", 0, 0),
            9:  PhysicsNode(9, "Cooling", "fluid", 3, 2),
            10: PhysicsNode(10, "Bypass", "fluid", 1, 1),
            11: PhysicsNode(11, "Fuel", "fluid", 0, 0),
            12: PhysicsNode(12, "Snsr.A", "sensor", 99, 99),
            13: PhysicsNode(13, "Snsr.B", "sensor", 99, 99),
        }
        self.nodes = cmapss_nodes
        
        # Hard forbidden structural rules (e.g., Snsr.B -> Fan)
        self.forbidden_edges.add((13, 0))
        self.forbidden_edges.add((12, 4))
        self.forbidden_edges.add((10, 3)) # Bypass cannot cause Combustor

    def evaluate_edge(self, src: int, dst: int, current_confidence: float) -> EvaluationResult:
        self.total_evaluations += 1
        violations = []
        score = 1.0
        
        src_node = self.nodes.get(src)
        dst_node = self.nodes.get(dst)
        
        if not src_node or not dst_node:
            return EvaluationResult(True, 1.0, current_confidence, [], "Unknown nodes, skipping physics.")

        # 1. Domain Forbidden Registry
        if (src, dst) in self.forbidden_edges:
            violations.append(ConstraintType.DOMAIN_FORBIDDEN.value)
            score *= 0.0
            
        # 2. Sensor Implausibility: Sensors measure, they do not causally influence physical upstream states
        if src_node.component_type == "sensor" and dst_node.component_type != "sensor":
            violations.append(ConstraintType.SENSOR_IMPLAUSIBILITY.value)
            score *= 0.0
            
        # 3. Thermodynamic Directionality: Exhaust cannot precede intake
        if src_node.component_type in ["turbine", "combustor"] and dst_node.component_type == "compressor":
            # EXCEPT mechanical spool connections (e.g. Turbine -> Spool -> Compressor)
            if src_node.id != 4 or dst_node.id != 2: # Ignore specific mechanical links if needed
                violations.append(ConstraintType.THERMODYNAMIC_DIRECTION.value)
                score *= 0.1
                
        # 4. Temperature Propagation: Cold fluid cannot spontaneously heat hot stages without combustion
        if src_node.component_type == "fluid" and src_node.temperature_tier < dst_node.temperature_tier:
            if dst_node.component_type == "combustor":
                pass # fluid can enter combustor
            else:
                violations.append(ConstraintType.CONSERVATION_VIOLATION.value)
                score *= 0.2

        is_valid = score > 0.3
        
        if not is_valid:
            self.total_violations += 1
            expl = f"Rejected edge {src_node.name} -> {dst_node.name} due to {', '.join(violations)}."
            self.rejected_edges.append({
                "src": src_node.name,
                "dst": dst_node.name,
                "reason": violations[0],
                "confidence": current_confidence
            })
            if len(self.rejected_edges) > 50:
                self.rejected_edges.pop(0)
        else:
            expl = f"Edge {src_node.name} -> {dst_node.name} passed thermodynamic constraints."

        # Compute uncertainty-aware confidence (discount confidence if physical score is borderline)
        adjusted_confidence = current_confidence * score

        return EvaluationResult(
            is_valid=is_valid,
            score=score,
            confidence=adjusted_confidence,
            violations=violations,
            explanation=expl
        )

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "total_violations": self.total_violations,
            "satisfaction_score": (self.total_evaluations - self.total_violations) / max(1, self.total_evaluations),
            "recent_rejections": self.rejected_edges[-5:] if self.rejected_edges else []
        }
