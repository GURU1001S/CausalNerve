import os
import csv
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

@dataclass
class InterventionRecord:
    intervention_id: str
    engine_id: str
    cycle_applied: int
    timestamp: str
    proposed_edge: Tuple[int, int]
    removed_edge: Optional[Tuple[int, int]]
    
    # State before intervention
    leakage_before: float
    lyapunov_before: float
    ece_before: float
    
    # State immediately after intervention
    leakage_after: float
    lyapunov_after: float
    ece_after: float
    
    # Long horizon tracking (cycles elapsed: value)
    leakage_trajectory: Dict[int, float] = field(default_factory=dict)
    lyapunov_trajectory: Dict[int, float] = field(default_factory=dict)
    
    # Performance metrics
    estimated_cycles_improvement: int = 0
    survival_duration: int = 0
    rollback_occurrence: bool = False
    environment_metadata: Dict[str, Any] = field(default_factory=dict)

class InterventionScore:
    """Computes physical metrics for long-horizon causal interventions."""
    
    @staticmethod
    def compute_structural_stability_gain(record: InterventionRecord) -> float:
        """Integral of Lyapunov decrease over survived horizon."""
        gain = 0.0
        v_base = record.lyapunov_before
        for cyc, v_t in record.lyapunov_trajectory.items():
            gain += (v_base - v_t)
        return gain

    @staticmethod
    def compute_causal_persistence(record: InterventionRecord, target_horizon: int = 100) -> float:
        """Fraction of time the leakage remained below baseline."""
        if not record.leakage_trajectory:
            return 0.0
        
        stable_points = sum(1 for v in record.leakage_trajectory.values() if v < record.leakage_before)
        return stable_points / len(record.leakage_trajectory)

    @staticmethod
    def compute_delayed_reward(record: InterventionRecord) -> float:
        """Aggregate ROI metric weighting stability and longevity."""
        persistence = InterventionScore.compute_causal_persistence(record)
        stability = InterventionScore.compute_structural_stability_gain(record)
        longevity = record.survival_duration / 500.0  # normalized against typical lifespan
        
        return (0.4 * persistence) + (0.4 * stability * 0.1) + (0.2 * longevity)

class SurvivalAnalysis:
    """Calculates fleet-wide survival probabilities for structural repairs."""
    
    @staticmethod
    def kaplan_meier_estimate(records: List[InterventionRecord]) -> Dict[int, float]:
        """
        Computes survival curve P(T > t) for repaired graphs.
        Returns mapping of cycle_duration -> survival_probability.
        """
        durations = [r.survival_duration for r in records if not r.rollback_occurrence]
        failures = [r.survival_duration for r in records if r.rollback_occurrence]
        
        if not durations and not failures:
            return {0: 1.0}
            
        all_times = sorted(list(set(durations + failures)))
        survival_curve = {}
        
        at_risk = len(records)
        surv_prob = 1.0
        
        for t in all_times:
            failed_at_t = sum(1 for f in failures if f == t)
            if at_risk > 0:
                surv_prob *= (1.0 - failed_at_t / at_risk)
            survival_curve[t] = surv_prob
            
            censored_at_t = sum(1 for d in durations if d == t)
            at_risk -= (failed_at_t + censored_at_t)
            
        return survival_curve

class AuditLayer:
    """Generates natural language physics-grounded explanations."""
    
    @staticmethod
    def generate_explanation(record: InterventionRecord) -> str:
        roi = InterventionScore.compute_delayed_reward(record)
        
        text = f"Surgical intervention on {record.proposed_edge} at cycle {record.cycle_applied} "
        if record.rollback_occurrence:
            text += f"failed structurally and was rolled back after {record.survival_duration} cycles. "
            text += f"Leakage returned to {record.leakage_before:.3f}."
        else:
            text += f"extended stable operation by +{record.survival_duration} cycles. "
            text += f"Leakage sustained a {max(0, record.leakage_before - record.leakage_after):.3f} reduction. "
            text += f"Long-horizon ROI score is {roi:.2f}."
            
        return text

class LongHorizonEvaluator:
    """
    Tracks and updates active interventions across multiple horizons (10, 50, 100, 300).
    """
    def __init__(self, storage_path: str = "intervention_memory.csv"):
        self.storage_path = storage_path
        self.lock = threading.RLock()
        self.active_records: Dict[str, InterventionRecord] = {}
        self.historical_records: List[InterventionRecord] = []
        self.tracking_horizons = [10, 50, 100, 300]
        
    def log_intervention(self, record: InterventionRecord):
        with self.lock:
            self.active_records[record.intervention_id] = record

    def update_trajectories(self, current_cycle: int, leakage: float, lyapunov: float, ece: float):
        with self.lock:
            completed = []
            for iid, record in self.active_records.items():
                elapsed = current_cycle - record.cycle_applied
                
                # Check predefined horizons
                for h in self.tracking_horizons:
                    if elapsed == h:
                        record.leakage_trajectory[elapsed] = leakage
                        record.lyapunov_trajectory[elapsed] = lyapunov
                        
                record.survival_duration = elapsed
                
                # Detect structural rollback / failure
                if leakage > record.leakage_before * 1.5 and elapsed > 5:
                    record.rollback_occurrence = True
                    completed.append(iid)
                
                # Stop tracking after max horizon
                if elapsed >= max(self.tracking_horizons):
                    completed.append(iid)
                    
            for iid in completed:
                rec = self.active_records.pop(iid)
                self.historical_records.append(rec)
                self.export_csv()

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        with self.lock:
            all_records = list(self.active_records.values()) + self.historical_records
            if not all_records:
                return {}
                
            roi_scores = [InterventionScore.compute_delayed_reward(r) for r in all_records]
            avg_roi = sum(roi_scores) / len(roi_scores) if roi_scores else 0
            
            surv_curve = SurvivalAnalysis.kaplan_meier_estimate(all_records)
            
            rollbacks = sum(1 for r in all_records if r.rollback_occurrence)
            rollback_prob = rollbacks / len(all_records) if all_records else 0.0
            
            avg_lifetime = sum(r.survival_duration for r in all_records) / len(all_records) if all_records else 0
            
            # Get latest explanation
            latest_audit = AuditLayer.generate_explanation(all_records[-1]) if all_records else ""
            
            return {
                "intervention_roi": round(avg_roi, 3),
                "repair_lifetime": round(avg_lifetime, 1),
                "rollback_probability": round(rollback_prob, 3),
                "survival_curve": surv_curve,
                "latest_audit": latest_audit,
                "total_interventions": len(all_records)
            }

    def export_csv(self):
        """Zero-dependency strict CSV export for reproducibility."""
        with self.lock:
            if not self.historical_records:
                return
                
            fields = list(self.historical_records[0].__dataclass_fields__.keys())
            try:
                with open(self.storage_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for r in self.historical_records:
                        d = asdict(r)
                        d['leakage_trajectory'] = json.dumps(d['leakage_trajectory'])
                        d['lyapunov_trajectory'] = json.dumps(d['lyapunov_trajectory'])
                        d['environment_metadata'] = json.dumps(d['environment_metadata'])
                        writer.writerow(d)
            except Exception as e:
                print(f"[LongHorizonEvaluator] Error exporting CSV: {e}")
