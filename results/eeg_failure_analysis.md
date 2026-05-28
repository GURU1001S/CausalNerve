# CausalNerve EEG Failure Analysis Report

## Executive Summary

While the `CausalNerve` framework demonstrated robust capabilities generalized from aerospace to neuroscience, the validation exposed several domain-specific failure modes when subjected to real physiological noise. This report openly documents these weaknesses to direct future iterations and prove our scientific rigor.

### 1. Hidden Confounder Failures
**Scenario:** In multiple windows, `CausalNerve` falsely inferred a direct causal link between `F3` and `F4` (interhemispheric frontal connection).
**Root Cause:** Subcortical structures (e.g., thalamus) commonly act as unobserved common parents driving both cortical regions simultaneously. Because the algorithm relies heavily on observed variables, unmeasured deep-brain confounders cause structural false positives.
**Implication:** The current `OCGR` (Online Causal Graph Revision) assumes causal sufficiency. In highly latent environments like EEG, this assumption is frequently violated.

### 2. Poor Observability Cases
**Scenario:** Graph topology resolution dropped significantly during periods of high-frequency gamma bursts where signal-to-noise ratio degraded.
**Root Cause:** Muscle artifacts (EMG) heavily contaminate higher frequencies. Without independent component analysis (ICA) integrated directly into the `CausalNerve` state vector, the causal leakage engine misinterpreted artifactual noise as structural divergence.
**Implication:** Need artifact rejection built into the `CausalTracer` or `Watch` loop.

### 3. Noise Sensitivity and Thresholding
**Scenario:** The `alarm_threshold` required recalibration from `0.05` (Aerospace) to `0.15` (EEG). 
**Root Cause:** Physiological data carries significantly more inherent variance (stochasticity) than deterministic thermodynamic systems. Using the aerospace threshold resulted in continuous "seizure-like" alarms that were merely biological baseline fluctuations.
**Implication:** `Lyapunov V(G)` energy gate must learn adaptive baselines instead of using static global thresholds.

### 4. Delayed-Effect Failures
**Scenario:** A visual stimulus occurred, but the occipital-to-frontal causal path was entirely missed by the single-step Markov assumption.
**Root Cause:** The `CounterfactualEngine` currently operates on immediate or next-step propagation. Neural signal propagation (e.g., P300 waves) spans hundreds of milliseconds (multiple windows). 
**Implication:** Multi-lag temporal causal discovery (e.g., VAR-based causal models) must be incorporated into `CausalNerve`.

### 5. Instability under Low Sampling Rates
**Scenario:** Sub-sampling the dataset to 64Hz resulted in complete collapse of the inferred causal graph (entropy peaked).
**Root Cause:** Many causal delays in the brain occur within 10-20ms. A 64Hz rate (~15ms resolution) suffers from severe aliasing of causal events, rendering them indistinguishable from simultaneous correlations.
**Implication:** `CausalNerve` is sensitive to Nyquist constraints not just for signal, but for *causal events*.

---
**Conclusion:** CausalNerve's math works across domains, but its fundamental assumptions (causal sufficiency, instantaneous propagation, high SNR) are stressed differently by biology compared to thermodynamics. Future versions will integrate FCI (Fast Causal Inference) rules for unobserved confounders to mitigate these issues.
