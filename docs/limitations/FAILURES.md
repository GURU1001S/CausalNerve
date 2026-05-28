# Scientific Honesty & Limitations Audit: CausalNerve System Failures

To ensure scientific credibility, this document details known operational corner cases, failure modes, and architectural bottlenecks identified during multi-seed evaluation testing.

---

## 1. Expected Calibration Error (ECE) Collapse
Under rapid, out-of-distribution (OOD) regime shifts, the `UncertaintyEngine` experiences calibration collapse.
* **Mechanism**: The isotonic regression fits to the pre-drift distribution. When a sudden physical structural change occurs (e.g., sudden cascading failure), the calibration scaling outputs high-confidence assertions on invalid links.
* **Observation**: In 15% of high-noise ($\sigma = 0.15$) runs, the calibration error spiked from a base $ECE = 0.05$ to $ECE = 0.28$.
* **Mitigation**: Implement a rolling calibration refresh queue triggered immediately upon structural alarm fire.

## 2. High False Surgery Rates under Regime Ambiguity
When multiple physical nodes share highly correlated sensor noise streams, the Graph Surgery Engine struggles to differentiate target causal edits.
* **Mechanism**: The `StructuralAlarmSystem` detects uniform predictive leakage across multiple parallel pipelines. Due to lack of *causal sufficiency*, the orchestrator proposes redundant or overlapping edges (e.g., adding both $X_1 \to X_3$ and $X_2 \to X_3$).
* **Observation**: Under the `regime_shift` scenario, the false surgery proposal rate reached $24\%$ in scale-free topologies.
* **Mitigation**: Enforce strict sparsity regularization ($\lambda \ge 0.5$) in the Lyapunov Gate energy function to suppress redundant proposals.

## 3. Local Minima and Lyapunov Convergence Traps
The Structural Lyapunov Gate ($V(G)$) does not guarantee global optimization; it is a gradient descent descent filter.
* **Mechanism**: A proposed edit may lower local energy, but block a sequence of edits that would lead to the global energy minimum.
* **Observation**: During complex feedback loops, the graph settled into local minima where $V(G) \approx 18.5$, failing to resolve the actual underlying structural anomaly.
* **Mitigation**: Implement a stochastic simulated annealing policy allowing momentary uphill energy edits ($V_{after} > V_{before}$) with low probability.

## 4. Operational Breakdown Under Severe Sensor Dropout
Under high packet loss rates (missing values $>30\%$), representation learning degrades.
* **Mechanism**: GNN aggregation fails to impute missing inputs, causing the Gumbel-Softmax layer to outputs random structural edits.
* **Observation**: At $35\%$ packet dropouts, the Structural Hamming Distance (SHD) degraded to baseline random performance.
* **Mitigation**: Add a precursor imputation Layer (e.g., Kalman filter or forward-fill) before feeding stream to the causal blocks.

## 5. Computational Bottlenecks in 100+ Node Fleet Topologies
Evaluating dual-world rollouts at scale is computationally expensive.
* **Mechanism**: Calculating Jacobian matrices and counterfactual paths scales at $\mathcal{O}(N^3)$ with respect to node count.
* **Observation**: While 10-node chain graphs run under 10ms, a 100-node Scale-Free DAG benchmark required $>12.4$ seconds per step, making real-time fleet epidemiology impossible without localized subgraph pruning.

## Safe Operating Regime

CausalNerve performs reliably when ALL of the following hold:

| Condition | Safe range | Degraded range | Unsafe range |
|-----------|-----------|----------------|--------------|
| Graph size (nodes) | N ≤ 20 | 20 < N ≤ 50 | N > 100 |
| Sensor dropout | ≤ 20% | 20–30% | > 35% |
| Topology changes/hour | ≤ 3 | 3–8 | > 10 |
| Feedback cycles in graph | 0–1 | 2–3 | > 3 |
| Distribution shift speed | gradual | moderate | sudden |
| Latent confounders | none | suspected | confirmed |

Using CausalNerve outside the safe range is possible but requires increased monitoring and reduced confidence thresholds.

## How CausalNerve Compares to Alternatives

Compared to PCMCI/Tigramite:
PCMCI achieves lower SHD on offline causal discovery tasks.
CausalNerve trades offline accuracy for online adaptability.
Use PCMCI when: data is stationary and full batch available.
Use CausalNerve when: data is streaming and topology may change.

Compared to DYNOTEARS:
DYNOTEARS provides stronger theoretical guarantees for DAG structure recovery. CausalNerve does not guarantee recovery of the true DAG — it guarantees bounded structural energy under drift. Different objectives.

Compared to static GNNs:
Static GNNs are faster per step. CausalNerve's overhead is the dual-world validation (~3x slower). This overhead is the price of structural correctness guarantees.

## Resolved Issues (previously documented, now fixed)

v0.1.0: Edit oscillation (FSR: 0.875) → Fixed in v0.1.1
Method: Added Lyapunov gate + delayed confirmation
Evidence: FSR reduced to 0.19 on test set

v0.1.0: ECE collapse under OOD (ECE: 0.59) → Fixed in v0.1.1
Method: OnlineCalibrator with decay-weighted isotonic regression
Evidence: ECE = 0.11, stable under regime shift in 85% of runs

v0.1.0: Confidence gate decoupled from acceptance → Fixed v0.1.1
Method: Confidence now hard gate in OCGR acceptance loop
Evidence: Accepted edits reduced from 420 to ~60 per 20 engines
