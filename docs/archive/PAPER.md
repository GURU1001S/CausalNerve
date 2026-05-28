# Online Causal Graph Revision with Lyapunov Stability

This document outlines the formal theoretical foundations of the CausalNerve framework.

## Abstract
Traditional causal discovery methods assume a static data generating process and output a fixed causal graph. In many real-world systems (e.g., degrading machinery, shifting climate patterns, adapting biological systems), the causal structure itself undergoes phase transitions. CausalNerve introduces the first mathematically grounded framework for Online Causal Graph Revision (OCGR) that is provably stable and validated through dual-world counterfactual simulations.

## 1. Sparse Causal Graph Engine (CSC/TICSC)
The core representation is a learned, sparse adjacency matrix updated via Gumbel-Softmax discretization. The engine operates in $O(N \cdot K)$ complexity where $N$ is the number of nodes and $K$ is the sparsity bound on parent nodes.

## 2. Online Causal Graph Revision (OCGR)
The graph revision process is driven by the detection of structural leakage — predictive error originating from mismatched causal assumptions.
When leakage exceeds threshold $\tau$, the system:
1. Localizes the anomalous edge via subset testing.
2. Proposes a minimal graph surgery (edge addition or deletion).
3. Validates the surgery via a "dual-world" forward simulation.

## 3. Intervention and Counterfactual Validation
Every structural change proposed by OCGR is subjected to an intervention test based on Pearl's $do(\cdot)$ operator.
A parallel "world model" is cloned, the proposed edge modification is applied, and the system simulates forward to verify that structural leakage $\mathcal{L}_{leak} \to 0$. Edges are only accepted if the counterfactual world demonstrates statistically significant improvement.

## 4. Lyapunov Stability Guarantee
Adaptive structural learning is highly susceptible to oscillation (flickering edges). CausalNerve solves this by framing graph edits as state transitions within a Lyapunov stability framework.
An energy function $V(G)$ is defined combining:
- Structural Leakage
- Graph Complexity (sparsity penalty)
- Transition Entropy (anti-oscillation penalty)

An edit $G \to G'$ is accepted if and only if:
$$\Delta V = V(G') - V(G) < 0$$
This guarantees monotonic convergence toward a stable structural sink.

## 5. Multi-Asset Structural Epidemiology
For systems deployed across fleets, CausalNerve tracks structural phase transitions as discrete trajectories. It uses Dynamic Time Warping (DTW) to compute structural similarities between assets and predicts upcoming graph revisions through nearest-neighbor trajectory voting.

## Limitations
- **Observational completeness:** The system assumes causal sufficiency for the defined node set; unobserved confounding variables can lead to false edge discovery if not handled via physics constraints.
- **Latency:** Dual-world rollout introduces a compute delay proportional to the rollout horizon, requiring batched or asynchronous validation loops for high-frequency streams.
