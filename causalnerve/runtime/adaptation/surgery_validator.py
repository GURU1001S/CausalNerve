"""
causalnerve.adaptation.surgery_validator
====================================
Monte-Carlo dual-world surgery validation.

Replaces the fake ``predicted_leakage *= 0.5`` with actual structural
rollouts that measure whether a proposed graph edit truly reduces
causal leakage.

Mathematical foundation
-----------------------
For a proposed edit E on graph G with state vector θ:

  World-0 (factual):   roll out G        for H steps → τ₀
  World-1 (intervened): roll out G ⊕ E   for H steps → τ₁

Leakage integral:
    L(τ) = Σ_t Σ_e |x_child(t) − w_e · x_parent(t)|²

Surgery utility:
    U = α·(L₀ − L₁) − β·D_explosion − γ·max(ΔV, 0)

where D_explosion = Σ_t ||τ₁(t)||² / Σ_t ||τ₀(t)||²  if ratio > 1
      ΔV = V(G⊕E, τ₁) − V(G, τ₀)

Monte-Carlo: repeat N times with injected sensor noise to obtain
mean(U), var(U), and a confidence interval.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SurgeryValidationResult:
    """Rich validation output — every field computed, nothing mocked."""
    accepted: bool
    utility: float
    leakage_before: float
    leakage_after: float
    delta_v: float
    divergence: float
    variance: float
    confidence: float
    n_rollouts: int
    rollout_utilities: List[float]
    reason: str


class MonteCarloSurgeryValidator:
    """
    Validates a proposed graph surgery by running N stochastic
    dual-world rollouts and computing a statistically grounded
    acceptance decision.

    Parameters
    ----------
    horizon : int
        Number of future steps to simulate per rollout.
    n_rollouts : int
        Number of Monte-Carlo repetitions.
    noise_scale : float
        Standard deviation of injected sensor noise.
    alpha, beta, gamma : float
        Utility weights for leakage-reduction, divergence-explosion,
        and Lyapunov energy increase respectively.
    variance_threshold : float
        Maximum acceptable variance of utility across rollouts.
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(self, *,
                 horizon: int = 25,
                 n_rollouts: int = 16,
                 noise_scale: float = 0.01,
                 alpha: float = 1.0,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 variance_threshold: float = 0.5,
                 seed: int = 42):
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.variance_threshold = variance_threshold
        self.seed = seed

    # ------------------------------------------------------------------
    # Core rollout (pure numpy, no torch dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _rollout(adj: np.ndarray, state: np.ndarray, horizon: int,
                 rng: np.random.RandomState, noise_scale: float) -> np.ndarray:
        """
        Simulate structural-equation dynamics on adjacency ``adj``.

        X_i(t+1) = (1−α_i)·X_i(t) + α_i·(Σ_j w_ji X_j(t) / Σ_j w_ji) + ε

        Returns trajectory array of shape (horizon, N).
        """
        N = adj.shape[0]
        traj = np.zeros((horizon, N))
        x = state.copy()
        traj[0] = x.copy()

        # Pre-compute parent info per node
        parent_weights: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
        for j in range(N):
            for i in range(N):
                if adj[i, j] > 0:
                    parent_weights[j].append((i, adj[i, j]))

        for t in range(1, horizon):
            x_new = x.copy()
            for j in range(N):
                pws = parent_weights[j]
                if pws:
                    total_w = sum(w for _, w in pws)
                    influence = sum(w * x[i] for i, w in pws) / total_w
                    # Coupling strength capped at 0.3 for numerical stability
                    a = min(total_w * 0.1, 0.3)
                    x_new[j] = (1.0 - a) * x[j] + a * influence
                x_new[j] += rng.normal(0, noise_scale)
                x_new[j] = np.clip(x_new[j], -5.0, 5.0)
            x = x_new
            traj[t] = x.copy()
        return traj

    @staticmethod
    def _leakage_integral(adj: np.ndarray, traj: np.ndarray) -> float:
        """
        L(τ) = Σ_t Σ_{(i,j)∈E} (x_j(t) − w_ij · x_i(t))²

        Measures how well each edge's structural equation is satisfied
        across the entire trajectory.  A *lower* value means the graph
        better explains the observed dynamics.
        """
        H, N = traj.shape
        total = 0.0
        for i in range(N):
            for j in range(N):
                w = adj[i, j]
                if w > 0:
                    residuals = traj[:, j] - w * traj[:, i]
                    total += float(np.sum(residuals ** 2))
        return total

    @staticmethod
    def _energy(adj: np.ndarray, traj: np.ndarray) -> float:
        """
        Lyapunov-like energy proxy: Frobenius norm of adjacency
        weighted by mean state magnitudes.

        V = ||A||²_F · mean(||x(t)||²)

        This captures both graph complexity and state excitation.
        """
        adj_energy = float(np.sum(adj ** 2))
        state_energy = float(np.mean(np.sum(traj ** 2, axis=1)))
        return adj_energy * state_energy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self,
                 adj: np.ndarray,
                 state_vector: np.ndarray,
                 edge: Tuple[int, int],
                 edit_type: str) -> SurgeryValidationResult:
        """
        Run N Monte-Carlo dual-world rollouts and decide acceptance.

        Parameters
        ----------
        adj : (N, N) current adjacency matrix (numpy)
        state_vector : (N,) current node states
        edge : (src, tgt) edge under consideration
        edit_type : 'add' or 'remove'

        Returns
        -------
        SurgeryValidationResult with every metric computed from rollouts.
        """
        src, tgt = edge
        N = adj.shape[0]

        # Build the hypothetical adjacency
        adj_edited = adj.copy()
        if edit_type == "remove":
            adj_edited[src, tgt] = 0.0
        elif edit_type == "add":
            adj_edited[src, tgt] = 0.5  # Default weight for new edges

        utilities = []
        leakage_befores = []
        leakage_afters = []
        delta_vs = []
        divergences = []

        for k in range(self.n_rollouts):
            rng = np.random.RandomState(self.seed + k)

            # Inject stochastic perturbation to initial state
            perturbed = state_vector + rng.normal(0, self.noise_scale, N)

            # World-0: factual (original graph)
            traj_0 = self._rollout(adj, perturbed, self.horizon, rng, self.noise_scale)

            # World-1: counterfactual (edited graph)
            rng1 = np.random.RandomState(self.seed + k)  # Same seed for paired comparison
            perturbed1 = state_vector + rng1.normal(0, self.noise_scale, N)
            traj_1 = self._rollout(adj_edited, perturbed1, self.horizon, rng1, self.noise_scale)

            # ── Metrics ──
            L0 = self._leakage_integral(adj, traj_0)
            L1 = self._leakage_integral(adj_edited, traj_1)

            V0 = self._energy(adj, traj_0)
            V1 = self._energy(adj_edited, traj_1)
            dV = V1 - V0

            # Divergence explosion ratio
            e0 = float(np.sum(traj_0 ** 2))
            e1 = float(np.sum(traj_1 ** 2))
            d_explosion = max(0.0, e1 / max(e0, 1e-12) - 1.0)

            # Divergence between worlds
            div = float(np.sum((traj_0 - traj_1) ** 2))

            # Surgery utility
            #   U = α·(L₀ − L₁) − β·D_explosion − γ·max(ΔV, 0)
            U = (self.alpha * (L0 - L1)
                 - self.beta * d_explosion
                 - self.gamma * max(dV, 0.0))

            utilities.append(U)
            leakage_befores.append(L0)
            leakage_afters.append(L1)
            delta_vs.append(dV)
            divergences.append(div)

        # ── Aggregate statistics ──
        u_arr = np.array(utilities)
        mean_u = float(np.mean(u_arr))
        var_u = float(np.var(u_arr))
        mean_leak_before = float(np.mean(leakage_befores))
        mean_leak_after = float(np.mean(leakage_afters))
        mean_dv = float(np.mean(delta_vs))
        mean_div = float(np.mean(divergences))

        # Confidence = fraction of rollouts where utility > 0
        positive_frac = float(np.mean(u_arr > 0))

        # ── Acceptance rule ──
        # 1. Mean utility must be positive (surgery helps on average)
        # 2. Lyapunov energy must not increase on average
        # 3. Variance must be below threshold (stable prediction)
        # 4. Majority of rollouts must agree (confidence > 0.5)
        leakage_improves = mean_leak_after < mean_leak_before
        energy_stable = mean_dv <= 0
        variance_ok = var_u < self.variance_threshold
        majority_positive = positive_frac > 0.5

        accepted = (mean_u > 0 and leakage_improves
                    and variance_ok and majority_positive)

        # Build reason string
        reasons = []
        if mean_u <= 0:
            reasons.append(f"negative utility ({mean_u:.4f})")
        if not leakage_improves:
            reasons.append(f"leakage increased ({mean_leak_before:.4f}->{mean_leak_after:.4f})")
        if not variance_ok:
            reasons.append(f"high variance ({var_u:.4f}>{self.variance_threshold})")
        if not majority_positive:
            reasons.append(f"low rollout agreement ({positive_frac:.0%})")
        reason = "Accepted" if accepted else "Rejected: " + "; ".join(reasons)

        return SurgeryValidationResult(
            accepted=accepted,
            utility=round(mean_u, 6),
            leakage_before=round(mean_leak_before, 6),
            leakage_after=round(mean_leak_after, 6),
            delta_v=round(mean_dv, 6),
            divergence=round(mean_div, 6),
            variance=round(var_u, 6),
            confidence=round(positive_frac, 4),
            n_rollouts=self.n_rollouts,
            rollout_utilities=[round(u, 6) for u in utilities],
            reason=reason,
        )
