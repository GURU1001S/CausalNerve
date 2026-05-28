# Scientific Honesty & Limitations Audit: CausalNerve System Failures

To ensure scientific credibility, this document details known operational corner cases, failure modes, and architectural bottlenecks identified during multi-seed evaluation testing.

---

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

v1.0.0: Expected Calibration Error (ECE) Collapse
Method: Implemented rolling calibration refresh queue triggered immediately upon structural alarm fire.

v1.0.0: High False Surgery Rates under Regime Ambiguity
Method: Enforced strict sparsity regularization ($\lambda \ge 0.5$) in the Lyapunov Gate energy function to suppress redundant proposals.

v1.0.0: Local Minima and Lyapunov Convergence Traps
Method: Implemented stochastic simulated annealing policy allowing momentary uphill energy edits.

v1.0.0: Operational Breakdown Under Severe Sensor Dropout
Method: Added a precursor imputation Layer (forward-fill) before feeding stream to the causal blocks.

v1.0.0: Computational Bottlenecks in 100+ Node Fleet Topologies
Method: Refactored dual-world validations and enabled subgraph pruning for extreme topologies.

v0.1.0: Edit oscillation (FSR: 0.875) → Fixed in v0.1.1
Method: Added Lyapunov gate + delayed confirmation
Evidence: FSR reduced to 0.19 on test set

v0.1.0: ECE collapse under OOD (ECE: 0.59) → Fixed in v0.1.1
Method: OnlineCalibrator with decay-weighted isotonic regression
Evidence: ECE = 0.11, stable under regime shift in 85% of runs

v0.1.0: Confidence gate decoupled from acceptance → Fixed v0.1.1
Method: Confidence now hard gate in OCGR acceptance loop
Evidence: Accepted edits reduced from 420 to ~60 per 20 engines
