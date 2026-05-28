import torch
import numpy as np
from typing import List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from scipy import stats

@dataclass
class SufficiencyResult:
    is_direct: bool
    p_value: float

class CausalSufficiencyChecker:
    """
    Before proposing edge (i->j), test whether i is a DIRECT cause of j
    or merely correlated via a common cause or mediator.
    """
    def __init__(self, state_history_window: int = 100, alpha: float = 0.05):
        self.window = state_history_window
        self.alpha = alpha

    def _partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        if z.shape[1] == 0:
            return stats.pearsonr(x, y)
        
        # Regress x on z
        beta_x = np.linalg.lstsq(z, x, rcond=None)[0]
        res_x = x - z @ beta_x
        
        # Regress y on z
        beta_y = np.linalg.lstsq(z, y, rcond=None)[0]
        res_y = y - z @ beta_y
        
        # Add tiny noise to avoid constant arrays
        res_x += np.random.normal(0, 1e-8, len(res_x))
        res_y += np.random.normal(0, 1e-8, len(res_y))
        
        return stats.pearsonr(res_x, res_y)

    def is_direct_cause(self, src: int, dst: int, state_history: torch.Tensor, conditioning_nodes: List[int]) -> SufficiencyResult:
        if state_history.shape[0] < 10:
            return SufficiencyResult(True, 0.0)
            
        history = state_history[-self.window:].cpu().numpy()
        if history.ndim == 3:
            history = history[:, 0, :] # (T, B, D) -> (T, D) if batch=1
        
        x = history[:, src]
        y = history[:, dst]
        
        if len(conditioning_nodes) > 0:
            z = history[:, conditioning_nodes]
            z = np.hstack([np.ones((z.shape[0], 1)), z])
        else:
            z = np.empty((history.shape[0], 0))
            
        r, p = self._partial_correlation(x, y, z)
        
        is_direct = p < self.alpha
        return SufficiencyResult(is_direct, float(p))

    def filter_proposals(self, proposals: List[Any], state_history: torch.Tensor, current_adj: torch.Tensor) -> List[Any]:
        filtered = []
        for p in proposals:
            if p.edit_type in ["add", "add_or_remove"]:
                src, dst = p.edge
                parents = torch.where(current_adj[:, dst] > 0.01)[0].tolist()
                cond_nodes = [k for k in parents if k != src]
                
                res = self.is_direct_cause(src, dst, state_history, cond_nodes)
                if res.is_direct:
                    filtered.append(p)
            else:
                filtered.append(p)
        return filtered

class DelayedConfirmationGate:
    """
    Multi-Tier Confirmation & Temporal Evidence Accumulation.
    Instead of fixed 3-cycle confirmation, requires N confirmations
    based on initial confidence. Failed validations reduce evidence
    instead of triggering hard resets.
    """
    def __init__(self, base_confirm: int = 3):
        self.base_confirm = base_confirm
        self.pending = {} 
        self.next_id = 0
        
    def _get_required_confirmations(self, confidence: float) -> int:
        if confidence >= 0.90: return 1
        elif confidence >= 0.75: return 2
        elif confidence >= 0.60: return 3
        else: return 5
        
    def submit(self, proposal: Any, initial_val_result: Any, confidence: float = 0.5) -> str:
        cid = str(self.next_id)
        self.next_id += 1
        req = self._get_required_confirmations(confidence)
        self.pending[cid] = {
            'proposal': proposal, 
            'evidence': 1.0, 
            'required': req,
            'last_val': initial_val_result,
            'cycles_alive': 0
        }
        return cid
        
    def step(self, validation_fn: Callable[[Any], Tuple[bool, Any]]) -> List[Tuple[Any, Any]]:
        confirmed = []
        rejected = []
        for cid in list(self.pending.keys()):
            item = self.pending[cid]
            item['cycles_alive'] += 1
            
            passed, val_res = validation_fn(item['proposal'])
            
            if passed:
                # Accumulate evidence
                item['evidence'] += 1.0
                item['last_val'] = val_res
            else:
                # Temporal penalty instead of hard reset
                item['evidence'] *= 0.5 
                
            if item['evidence'] >= item['required']:
                confirmed.append((item['proposal'], item['last_val']))
                del self.pending[cid]
            elif item['evidence'] < 0.25 or item['cycles_alive'] > 10:
                rejected.append(item['proposal'])
                del self.pending[cid]
                
        return confirmed

class AdaptiveAlarmThreshold:
    """
    Alarm threshold tightens after a recent false alarm
    and relaxes after a period of structural stability.
    """
    def __init__(self,
                  base_threshold: float = 0.05,
                  false_alarm_penalty: float = 2.0,
                  stability_reward: float = 0.95,
                  min_threshold: float = 0.02,
                  max_threshold: float = 0.20):
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.false_alarm_penalty = false_alarm_penalty
        self.stability_reward = stability_reward
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cycles_since_alarm = 0
        
    def update(self, alarm_fired: bool, edit_was_accepted: bool, edit_was_correct: Optional[bool] = None):
        if alarm_fired:
            self.cycles_since_alarm = 0
            if not edit_was_accepted:
                self.threshold = min(self.max_threshold, self.threshold * self.false_alarm_penalty)
            elif edit_was_correct is True:
                self.threshold = max(self.min_threshold, self.threshold * 0.9)
        else:
            self.cycles_since_alarm += 1
            if self.cycles_since_alarm > 20:
                self.threshold = max(self.min_threshold, self.threshold * self.stability_reward)
                
    def current(self) -> float:
        return self.threshold
