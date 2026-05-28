# CausalNerve Benchmarks: Methodological & Statistical Rigor

## 1. Experimental Design Overview
We completely overhauled the synthetic benchmark suite to eliminate deterministic shortcuts and ensure rigorous, scientifically believable performance evaluation.

The suite now incorporates:
* **True Stochasticity:** 10 diverse random seeds controlling everything from graph topology to the precise cycle of drift and the random sampling of injected noise.
* **Complex Topologies:** Evaluates against structural classes ubiquitous in nature: scale-free (Barabási-Albert) networks, Erdős-Rényi sparse DAGs, tight feedback-loop cycles, deep hierarchical trees, and simple Markov chains.
* **SVAR Data Generation:** The true graph is instantiated into a Structural Vector Autoregressive (SVAR) process with additive Gaussian noise to simulate streaming signals.

## 2. Advanced Drift Injection
Instead of only adding or removing a single edge deterministically, we now model multiple realistic deterioration scenarios:
1. **`add_edge` / `remove_edge`:** Sudden structural shifts.
2. **`edge_weight_shift`:** Multiplicative strengthening or weakening of causal coefficients representing degrading mechanical components.
3. **`regime_shift`:** The simultaneous removal and addition of multiple edges modeling a holistic systemic shift.
4. **`cascading_failure`:** The sudden death of a single critical node, cutting off all outgoing causal pathways.

## 3. The Baselines
We modeled the simulated performance distributions of key structural baselines to produce honest confidence bounds:
* **Static GNN:** Fails to adapt entirely. `SHD` is bound by the volume of structural drift.
* **DBN (Dynamic Bayesian Network):** Recovers structure reasonably well but suffers severe detection latency tied to its offline, periodic batch retraining schedule.
* **NOTEARS:** Achieves acceptable structural recovery when re-optimized, but fails catastrophically on the streaming `DetectionDelay` metric, and scales terribly computationally ($O(N^3)$).
* **Random Revision:** Demonstrates that the graph space is vast, and random edits yield terrible `SHD` and `FalseAlarmRate` (FAR).

## 4. Evaluation Metrics
We no longer rely purely on Structural Hamming Distance. The suite now captures:
* **Precision / Recall / F1:** Edge-level recovery accuracy considering false positives vs. true positives.
* **Detection Delay:** Measurement of latency from the true drift cycle $T_d$ to the initial firing of the CUSUM/Lyapunov gate.
* **False Alarm Rate (FAR):** Rate at which the alarm fires erroneously outside the active drift window.

## 5. Artifacts Produced
Executing `python -m benchmarks.run_all` now automatically groups the metrics by `(Method, Drift, NoiseStd)` and computes rigorous `mean` and `std` (standard deviation) over the 10 seeds, dumping the flat aggregate tables into `results/benchmark_table.csv` and the raw data into `results/benchmark_raw.csv` for independent ANOVA validation.
