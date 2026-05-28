import torch
from typing import Any, Tuple, List

class CalibrationStatus:
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    COLLAPSED = "COLLAPSED"

class CalibrationMonitor:
    """
    Monitors calibration quality in real time.
    Raises alarm before collapse propagates to bad decisions.
    """
    
    def __init__(self,
                  baseline_ece: float = 0.05,
                  collapse_threshold: float = 0.20,
                  window: int = 50):
        self.baseline_ece = baseline_ece
        self.collapse_threshold = collapse_threshold
        self.window = window
        self.stable_cycles = 0
        self.status = CalibrationStatus.HEALTHY
        
    def step(self, ece_current: float) -> str:
        if ece_current > max(4 * self.baseline_ece, self.collapse_threshold):
            self.status = CalibrationStatus.COLLAPSED
            self.stable_cycles = 0
        elif ece_current > 2 * self.baseline_ece:
            # If we are collapsed, we must stay collapsed until we recover
            if self.status != CalibrationStatus.COLLAPSED:
                self.status = CalibrationStatus.WARNING
            self.stable_cycles = 0
        else:
            self.stable_cycles += 1
            if self.stable_cycles >= self.window:
                self.status = CalibrationStatus.HEALTHY
                
        return self.status
    
    def conservative_fallback(self, edit_proposal: Any, thermo_consistent: bool, fleet_prior_score: float, leakage_delta: float) -> bool:
        """
        Used when calibration is COLLAPSED.
        Accept only if ALL three conditions hold simultaneously:
            1. leakage_delta > 0.20 (strong signal)
            2. thermo_consistent == True
            3. fleet_prior_score > 0.50
        """
        if leakage_delta > 0.20 and thermo_consistent and fleet_prior_score > 0.50:
            return True
        return False

class OODResult:
    def __init__(self, is_ood: bool, distance_score: float, ood_dims: List[int]):
        self.is_ood = is_ood
        self.distance_score = distance_score
        self.ood_dims = ood_dims

class OODDetector:
    """
    Detect when current state is outside the training distribution.
    If OOD: apply extra skepticism to all edit proposals.
    """
    
    def __init__(self, reference_states: torch.Tensor):
        # We use Mahalanobis distance conceptually, implemented efficiently
        # reference_states shape: (N_samples, D)
        if reference_states.dim() == 3:
            # (B, N, D) -> (B*N, D)
            reference_states = reference_states.view(-1, reference_states.shape[-1])
            
        self.mean = reference_states.mean(dim=0)
        # Compute covariance matrix
        centered = reference_states - self.mean
        self.cov = (centered.T @ centered) / (reference_states.shape[0] - 1)
        
        # Add small regularizer to diagonal for stability
        self.cov += torch.eye(self.cov.shape[0], device=self.cov.device) * 1e-4
        self.inv_cov = torch.linalg.inv(self.cov)
        
    def is_ood(self, current_state: torch.Tensor, threshold: float = 3.0) -> OODResult:
        if current_state.dim() == 3:
            current_state = current_state.view(-1, current_state.shape[-1])
            
        centered = current_state - self.mean
        # Mahalanobis distance squared
        dist_sq = (centered @ self.inv_cov * centered).sum(dim=1)
        dist = torch.sqrt(dist_sq).mean().item()
        
        is_ood = dist > threshold
        
        # Find which dimensions contributed most to distance
        contributions = (centered @ self.inv_cov * centered).mean(dim=0)
        ood_dims = torch.where(contributions > (threshold ** 2) / current_state.shape[-1])[0].tolist()
        
        return OODResult(is_ood=is_ood, distance_score=dist, ood_dims=ood_dims)
