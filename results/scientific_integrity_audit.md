# CausalNerve Scientific Integrity Audit

**Date:** 2026-05-26
**Objective:** Identify any remaining parts of the framework that fake outputs, use placeholders, hardcode metrics, simulate confidence improperly, or bypass true computation.

---

## 1. Reasoning Engine (`causalnerve/reason/`)
**Classification:** **REAL**

*   **Mathematical Validity:** High. Implements Pearl’s exact $do$-calculus via structural graph severing and topological propagation. Counterfactuals run true dual-world structural equation simulations with $L_2$ divergence tracking.
*   **Implementation Completeness:** Complete. Zero mocks remain.
*   **Reviewer Risk Level:** Low.
*   **Production Readiness:** Ready.

## 2. Public API / SDK (`causalnerve/sdk.py`)
**Classification:** **REAL**

*   **Mathematical Validity:** High. All endpoints (`why`, `what_if`, `do`, `run_counterfactual`, `watch`) delegate strictly to mathematical engines. Type validation and graph consistency checks are enforced.
*   **Implementation Completeness:** Complete.
*   **Reviewer Risk Level:** Low.
*   **Production Readiness:** Ready.

## 3. Probability Calibration (`causalnerve/adapt/calibrator.py`)
**Classification:** **REAL** (with **HEURISTIC** fallback)

*   **Mathematical Validity:** Strong. Uses standard Expected Calibration Error (ECE) and Isotonic Regression for rolling-window confidence calibration.
*   **Implementation Completeness:** High. Includes robust fallbacks if dependencies are missing.
*   **Reviewer Risk Level:** Low.
*   **Production Readiness:** Ready.

## 4. Graph Revision / OCGR (`causalnerve/adapt/ocgr.py`)
**Classification:** **PARTIAL / MOCK**

*   **Mathematical Validity:** Compromised. The orchestration (Lyapunov gating, artifact dropout filters) is structurally sound, but the actual proposal validation relies on hardcoded assumptions rather than simulation.
*   **Implementation Completeness:** Low. 
    *   *Finding 1:* `predicted_leakage[-1] *= 0.5` is hardcoded to blindly assume that any proposed edit automatically halves system leakage. 
    *   *Finding 2:* Bayesian Confidence Fusion is simulated with arbitrary static bumps (`fused_conf += 0.15` for priority).
*   **Reviewer Risk Level:** **CRITICAL**. Academic reviewers will immediately flag the simulated 50% leakage reduction as scientifically dishonest.
*   **Production Readiness:** Not Ready. `GraphSurgeryEngine.validate()` must be wired to use the `CounterfactualEngine` to physically test edits.

## 5. Fleet Epidemiology (`causalnerve/fleet/epidemiology.py`)
**Classification:** **MOCK / HEURISTIC**

*   **Mathematical Validity:** Poor.
*   **Implementation Completeness:** Very Low. It functions purely as a tracking dictionary with fake math.
    *   *Finding 1:* `dtw_match()` strictly returns a hardcoded `0.95`.
    *   *Finding 2:* `compute_fleet_stability_index()` normalizes using an arbitrary hardcoded heuristic: `max(0.0, 1.0 - (avg_motifs / 5.0))`.
    *   *Finding 3:* `TransferLearningLayer` uses naive counting without proper causal confounding adjustment.
*   **Reviewer Risk Level:** **CRITICAL**. Hardcoded similarities and arbitrary division scales invalidate the fleet intelligence claims.
*   **Production Readiness:** Not Ready. Requires true Dynamic Time Warping (DTW) implementations and statistically grounded stability indices.

---

### Audit Summary & Required Actions

To ensure CausalNerve is scientifically honest prior to publication, the following critical rewrites are required:

1.  **Rewrite OCGR Edit Validation:** Remove the `predicted_leakage *= 0.5` cheat. The surgery engine must query the `CounterfactualEngine` to see if removing the edge actually reduces structural leakage.
2.  **Implement Real DTW in Epidemiology:** Replace `return 0.95` with an actual Dynamic Time Warping algorithm using `scipy` or `fastdtw` to compare anomaly motif trajectories.
3.  **Fix Fleet Stability Index:** Derive the stability metric from actual graph entropy or Lyapunov energy states, not arbitrary division.
