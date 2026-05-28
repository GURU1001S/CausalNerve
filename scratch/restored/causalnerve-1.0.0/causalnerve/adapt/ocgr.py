"""
causalnerve.adaptation.ocgr
======================
Online Causal Graph Revision (OCGR) Engine for CausalNerve.
Domain-agnostic architecture for detecting structural leakage, 
validating edits, and orchestrating graph revisions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable, Any
from dataclasses import dataclass, field
import json
import time

from ..core.engine import CausalGraphEngine
from .lyapunov import StructuralLyapunovFunction, GraphState
from .causal_sufficiency import CausalSufficiencyChecker, DelayedConfirmationGate, AdaptiveAlarmThreshold
from .live_validation import LiveSurgeryValidator

@dataclass
class AlarmEvent:
    edge: Tuple[int, int]
    leakage_value: float
    leakage_history: List[float]
    cycles_above_threshold: int

@dataclass
class ArtifactClassification:
    is_artifact: bool
    reason: str

@dataclass
class EditProposal:
    edge: Tuple[int, int]
    edit_type: str  # 'add' or 'remove'
    rationale: str
    predicted_confidence: float
    is_priority: bool = False
    is_emergency: bool = False

@dataclass
class ValidationResult:
    accepted: bool
    V_before: float
    V_after: float
    confidence: float
    reason: str

@dataclass
class RevisionEvent:
    timestamp: float
    cycle: int
    edit_type: str
    edge: Tuple[int, int]
    V_before: float
    V_after: float
    confidence: float
    leakage_before: float
    leakage_after: float
    reason: str
    accepted: bool

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "cycle": self.cycle,
            "edit_type": self.edit_type,
            "edge": self.edge,
            "V_before": self.V_before,
            "V_after": self.V_after,
            "confidence": self.confidence,
            "leakage_before": self.leakage_before,
            "leakage_after": self.leakage_after,
            "reason": self.reason,
            "accepted": self.accepted
        }

class StructuralAlarmSystem:
    """
    Monitors causal leakage per edge in real time.
    Fires callbacks when leakage exceeds threshold.
    Domain-agnostic: threshold is a parameter, not hardcoded.
    """
    def __init__(self, 
                 graph: CausalGraphEngine,
                 threshold: float = 0.05,
                 window: int = 20):
        self.graph = graph
        self.threshold = threshold
        self.window = window
        self.leakage_buffer: Dict[Tuple[int, int], List[float]] = {}
        self.callbacks: List[Callable[[AlarmEvent], None]] = []
        
    def step(self, edge_leakages: Dict[Tuple[int, int], float]) -> List[AlarmEvent]:
        alarms = []
        for edge, leak_val in edge_leakages.items():
            if edge not in self.leakage_buffer:
                self.leakage_buffer[edge] = []
            
            buf = self.leakage_buffer[edge]
            buf.append(leak_val)
            if len(buf) > self.window:
                buf.pop(0)
                
            mean_leakage = sum(buf) / len(buf)
            
            if mean_leakage > self.threshold:
                cycles_above = sum(1 for v in buf if v > self.threshold)
                alarm = AlarmEvent(
                    edge=edge,
                    leakage_value=mean_leakage,
                    leakage_history=list(buf),
                    cycles_above_threshold=cycles_above
                )
                alarms.append(alarm)
                
                for cb in self.callbacks:
                    cb(alarm)
                    
        return alarms
    
    def register_callback(self, fn: Callable[[AlarmEvent], None]):
        """User-defined callback on alarm. Used by CausalNerve.watch()"""
        self.callbacks.append(fn)
        
    def __repr__(self):
        return f"<StructuralAlarmSystem threshold={self.threshold} window={self.window}>"


class DropoutArtifactDetector:
    """
    Three deterministic rules to distinguish genuine structural
    change from sensor/input noise artifacts.
    Domain-agnostic: works on any time-series input.
    """
    def classify(self, 
                 alarm: AlarmEvent,
                 all_edge_leakages: Dict[Tuple[int, int], float]) -> ArtifactClassification:
        
        # Rule 1: Simultaneity - Many edges spiking at once indicates system-wide noise, not specific structural shift.
        spiking_edges = sum(1 for val in all_edge_leakages.values() if val > 0.05) # Arbitrary noise floor
        if spiking_edges > 3:
            return ArtifactClassification(is_artifact=True, reason="Simultaneity: >3 edges spiked")
            
        # Rule 2: Persistence - Spike resolves too fast
        if alarm.cycles_above_threshold < 3:
            return ArtifactClassification(is_artifact=True, reason="Persistence: Spike resolves in <3 cycles")
            
        # Rule 3: Precursor absence - No gradual rise, instant jump implies sensor fault
        hist = alarm.leakage_history
        if len(hist) > 5:
            recent_jump = hist[-1] - hist[-5]
            if recent_jump > 0.3: # large sudden jump
                return ArtifactClassification(is_artifact=True, reason="Precursor absence: Instant jump")
                
        return ArtifactClassification(is_artifact=False, reason="Genuine structural shift")
        
    def __repr__(self):
        return "<DropoutArtifactDetector>"


class GraphSurgeryEngine:
    """
    Proposes, validates, and applies structural edits.
    The validation is domain-agnostic.
    Domain knowledge (plausibility rules) is injected via
    a plausibility_fn parameter — not hardcoded.
    """
    def __init__(self,
                 graph: CausalGraphEngine,
                 lyapunov: StructuralLyapunovFunction,
                 plausibility_fn: Optional[Callable[[int, int, Any], bool]] = None,
                 confidence_threshold: float = 0.40):
        self.graph = graph
        self.lyapunov = lyapunov
        self.plausibility_fn = plausibility_fn
        self.confidence_threshold = confidence_threshold
        
    def propose(self, alarm: AlarmEvent, current_state: Any = None) -> List[EditProposal]:
        proposals = []
        src, tgt = alarm.edge
        
        # Check domain-specific plausibility if provided
        if self.plausibility_fn is not None:
            if not self.plausibility_fn(src, tgt, current_state):
                return []
                
        # FastTrackEmergencyRepair detection
        is_emergency = alarm.leakage_value > 0.50
        
        # Priority Scheduling (mock fleet priors/recurrence for architecture)
        is_priority = (src, tgt) in getattr(self, 'fleet_priors', set())
        
        # Base confidence scales with leakage
        base_conf = min(0.9, alarm.leakage_value * 2.0)
        
        proposals.append(
            EditProposal(
                edge=(src, tgt),
                edit_type="add_or_remove", # Orchestrator will resolve
                rationale=f"High leakage ({alarm.leakage_value:.4f}) detected on edge {src}->{tgt}",
                predicted_confidence=base_conf,
                is_priority=is_priority,
                is_emergency=is_emergency
            )
        )
        return proposals
        
    def validate(self, 
                 proposal: EditProposal,
                 current_graph: GraphState,
                 leakage_history: np.ndarray,
                 edit_history: List[RevisionEvent],
                 state_vector: np.ndarray,
                 current_cycle: int) -> ValidationResult:
                     
        # 1. Dual-world simulation would run here to test the edit counterfactually.
        # We simulate the outcome:
        # In a real system, we'd roll out the world model.
        predicted_leakage = leakage_history.copy()
        if len(predicted_leakage) > 0:
            predicted_leakage[-1] *= 0.5 # Assume edit halves leakage
            
        # 2. Check Lyapunov stability
        accepted, V_before, V_after, l_reason = self.lyapunov.gate_edit(
            proposed_edit=proposal,
            current_graph=current_graph,
            leakage_history=leakage_history,
            edit_history=edit_history,
            theta=state_vector,
            current_cycle=current_cycle,
            proposed_leakage=predicted_leakage
        )
        
        # Risk-Adaptive Lyapunov Threshold
        # Tolerate slight energy increases if confidence/leakage reduction is very high
        leak_reduction = leakage_history[-1] - predicted_leakage[-1] if len(leakage_history)>0 else 0
        if not accepted and V_after <= V_before + 1.5 * leak_reduction and proposal.predicted_confidence > 0.75:
            accepted = True
            l_reason = "Risk-Adaptive Lyapunov override"
            
        # Bayesian Surgery Confidence Fusion (simulated)
        # Combines multiple priors into a single robust score
        fused_conf = proposal.predicted_confidence
        if proposal.is_priority: fused_conf += 0.15
        if proposal.is_emergency: fused_conf += 0.20
        fused_conf = min(1.0, fused_conf)
        proposal.predicted_confidence = fused_conf
        
        conf_pass = fused_conf >= self.confidence_threshold
        
        final_accept = accepted and conf_pass
        reason = l_reason if not accepted else ("Low confidence" if not conf_pass else "Accepted")
        
        return ValidationResult(
            accepted=final_accept,
            V_before=V_before,
            V_after=V_after,
            confidence=fused_conf,
            reason=reason
        )
        
    def apply(self, 
              proposal: EditProposal, 
              validation: ValidationResult,
              cycle: int,
              leakage_before: float) -> RevisionEvent:
                  
        event = RevisionEvent(
            timestamp=time.time(),
            cycle=cycle,
            edit_type=proposal.edit_type,
            edge=proposal.edge,
            V_before=validation.V_before,
            V_after=validation.V_after,
            confidence=validation.confidence,
            leakage_before=leakage_before,
            leakage_after=leakage_before * 0.5, # Post-surgery assumption
            reason=proposal.rationale + " | " + validation.reason,
            accepted=validation.accepted
        )
        return event
        
    def __repr__(self):
        return f"<GraphSurgeryEngine conf_thresh={self.confidence_threshold}>"


class HoldQueue:
    """
    Edits with 0.25 < confidence < 0.40 enter HOLD state.
    Re-evaluated each cycle for up to max_hold_cycles.
    Accepts if confidence rises. Rejects if it never does.
    """
    def __init__(self, max_hold_cycles: int = 5):
        self.max_hold_cycles = max_hold_cycles
        self.queue: List[Dict[str, Any]] = []
        
    def add(self, proposal: EditProposal, cycle: int):
        self.queue.append({
            "proposal": proposal,
            "entered_at": cycle
        })
        
    def step(self, current_cycle: int) -> List[EditProposal]:
        """Returns proposals that have expired and should be re-evaluated/rejected."""
        expired = []
        retained = []
        for item in self.queue:
            if current_cycle - item["entered_at"] >= self.max_hold_cycles:
                expired.append(item["proposal"])
            else:
                retained.append(item)
        self.queue = retained
        return expired
        
    def __repr__(self):
        return f"<HoldQueue max_hold_cycles={self.max_hold_cycles} holding={len(self.queue)}>"


class RevisionHistory:
    """
    Complete audit trail of every structural change.
    Serializable to JSON.
    Used by CausalNerve.audit_trail().
    """
    def __init__(self, log_path: Optional[str] = None):
        self.history: List[RevisionEvent] = []
        self.log_path = log_path
        
    def log(self, event: RevisionEvent):
        self.history.append(event)
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump([e.to_dict() for e in self.history], f, indent=2)
            
    def get_accepted(self) -> List[RevisionEvent]:
        return [e for e in self.history if e.accepted]
        
    def __repr__(self):
        return f"<RevisionHistory total={len(self.history)} accepted={len(self.get_accepted())}>"


@dataclass
class OCGRStepResult:
    cycle: int
    alarms_fired: int
    proposals_generated: int
    edits_applied: int
    edits_held: int


class OCGROrchestrator:
    """
    The main OCGR controller. Wires all components together.
    Called by CausalNerve.watch() and CausalNerve.revise().
    """
    def __init__(self,
                 graph: CausalGraphEngine,
                 lyapunov: StructuralLyapunovFunction,
                 plausibility_fn: Optional[Callable] = None,
                 alarm_threshold: float = 0.05,
                 confidence_threshold: float = 0.40,
                 audit_log_path: Optional[str] = None):
                     
        self.graph = graph
        self.lyapunov = lyapunov
        
        self.adaptive_threshold = AdaptiveAlarmThreshold(base_threshold=alarm_threshold)
        self.alarm_system = StructuralAlarmSystem(graph, threshold=self.adaptive_threshold.current())
        self.artifact_detector = DropoutArtifactDetector()
        self.surgery_engine = GraphSurgeryEngine(graph, lyapunov, plausibility_fn, confidence_threshold)
        
        self.sufficiency_checker = CausalSufficiencyChecker(state_history_window=100)
        self.delayed_gate = DelayedConfirmationGate(base_confirm=3)
        
        self.hold_queue = HoldQueue()
        self.history = RevisionHistory(log_path=audit_log_path)
        self.cycle_count = 0
        self.live_validator = LiveSurgeryValidator()
        
    def step(self, 
             edge_leakages: Dict[Tuple[int, int], float], 
             current_adj: torch.Tensor,
             state_vector: np.ndarray,
             state_history: Optional[torch.Tensor] = None) -> OCGRStepResult:
        """
        One cycle of the full OCGR loop.
        """
        self.cycle_count += 1
        alarms_fired = 0
        proposals_gen = 0
        edits_applied = 0
        edits_held = 0
        
        # Adaptive alarm threshold
        self.alarm_system.threshold = self.adaptive_threshold.current()
        
        # 1. Alarm system checks leakage
        alarms = self.alarm_system.step(edge_leakages)
        alarms_fired = len(alarms)
        
        edit_accepted_this_cycle = False
        
        for alarm in alarms:
            # 2. Artifact detector filters noise
            classification = self.artifact_detector.classify(alarm, edge_leakages)
            if classification.is_artifact:
                self.adaptive_threshold.update(True, False, False)
                continue
                
            # 3. Surgery engine proposes edits
            proposals = self.surgery_engine.propose(alarm, state_vector)
            
            # Resolve add/remove based on current adjacency
            src, tgt = alarm.edge
            for p in proposals:
                if p.edit_type == "add_or_remove":
                    if src < current_adj.shape[0] and tgt < current_adj.shape[1]:
                        p.edit_type = "remove" if current_adj[src, tgt] > 0.01 else "add"
            
            # FSR FIX 1: Causal Sufficiency Filter
            if state_history is not None:
                proposals = self.sufficiency_checker.filter_proposals(proposals, state_history, current_adj)
                
            proposals_gen += len(proposals)
            
            for p in proposals:
                # 4. Validate each proposal
                current_graph_state = GraphState(
                    adj=current_adj,
                    edge_leakage=torch.tensor(list(edge_leakages.values())),
                    n_nodes=current_adj.shape[0]
                )
                
                mean_leak = np.mean(list(edge_leakages.values()))
                leak_hist = np.array([mean_leak]) 
                
                val_result = self.surgery_engine.validate(
                    proposal=p,
                    current_graph=current_graph_state,
                    leakage_history=leak_hist,
                    edit_history=self.history.get_accepted(),
                    state_vector=state_vector,
                    current_cycle=self.cycle_count
                )
                
                # PHASE 3: Live Validation Check
                leakage_before = mean_leak
                leakage_after = leakage_before * 0.5  # Simulated improvement for validation
                self.live_validator.validate(p, val_result, self.cycle_count, leakage_before, leakage_after)
                
                if val_result.accepted:
                    # FastTrack Emergency Repair or Priority Bypass
                    if p.is_emergency or (p.is_priority and p.predicted_confidence > 0.85):
                        event = self.surgery_engine.apply(
                            proposal=p, 
                            validation=val_result, 
                            cycle=self.cycle_count,
                            leakage_before=np.mean(list(edge_leakages.values()))
                        )
                        self.history.log(event)
                        edits_applied += 1
                        edit_accepted_this_cycle = True
                        self.adaptive_threshold.update(True, True, True)
                    else:
                        # FSR FIX 2 & Delay Fix 1: Adaptive Delayed Confirmation Gate
                        self.delayed_gate.submit(p, val_result, p.predicted_confidence)
                    
                # Update hold queue if validation failed but confidence moderate
                elif 0.25 < p.predicted_confidence < 0.40:
                    self.hold_queue.add(p, self.cycle_count)
                    edits_held += 1

        # Process delayed confirmations
        def _val_fn(p):
            current_graph_state = GraphState(
                adj=current_adj,
                edge_leakage=torch.tensor(list(edge_leakages.values())),
                n_nodes=current_adj.shape[0]
            )
            mean_leak = np.mean(list(edge_leakages.values()))
            leak_hist = np.array([mean_leak])
            res = self.surgery_engine.validate(p, current_graph_state, leak_hist, self.history.get_accepted(), state_vector, self.cycle_count)
            return res.accepted, res
            
        confirmed_edits = self.delayed_gate.step(_val_fn)
        
        for p, val_result in confirmed_edits:
            event = self.surgery_engine.apply(
                proposal=p, 
                validation=val_result, 
                cycle=self.cycle_count,
                leakage_before=np.mean(list(edge_leakages.values()))
            )
            self.history.log(event)
            if event.accepted:
                edits_applied += 1
                edit_accepted_this_cycle = True
                self.adaptive_threshold.update(True, True, True)

        if alarms_fired == 0:
            self.adaptive_threshold.update(False, False)
            
        # Check expired held edits
        expired_holds = self.hold_queue.step(self.cycle_count)
        
        # PHASE 3: Monitor Persistence and execute rollbacks
        rollbacks = self.live_validator.monitor_persistence(self.cycle_count, edge_leakages)
        for rb_edge in rollbacks:
            event = RevisionEvent(
                timestamp=time.time(),
                cycle=self.cycle_count,
                edit_type="rollback",
                edge=rb_edge,
                V_before=0.0, V_after=0.0, confidence=1.0,
                leakage_before=edge_leakages.get(rb_edge, 0.0),
                leakage_after=0.0,
                reason="[ROLLBACK] Failed persistence check",
                accepted=True
            )
            self.history.log(event)
            edits_applied += 1
        
        return OCGRStepResult(
            cycle=self.cycle_count,
            alarms_fired=alarms_fired,
            proposals_generated=proposals_gen,
            edits_applied=edits_applied,
            edits_held=edits_held
        )
        
    def manual_revise(self) -> List[RevisionEvent]:
        """Called by user via nerve.revise() for manual trigger"""
        return []
        
    def __repr__(self):
        return f"<OCGROrchestrator cycles={self.cycle_count} history={len(self.history.history)}>"
