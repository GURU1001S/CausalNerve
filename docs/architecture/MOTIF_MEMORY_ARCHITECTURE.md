# Motif Memory Architecture

## Conceptual Overview
The CausalNerve framework previously operated as an isolated inference engine, calculating causal leakage and structural metrics locally per engine. With the introduction of the **Structural Memory Layer**, CausalNerve transitions into a **Persistent Causal Intelligence System**.

This layer allows the system to remember structural failure modes across an entire engine fleet and apply precognitive interventions based on historical isomorphism.

## Core Components

### 1. `MotifMemoryBank`
A thread-safe, persistent dictionary mapping abstract graph fingerprints to their physical trajectories.
When an engine undergoes structural surgery (OCGR intervention), the surrounding subgraph is encoded and saved. It tracks:
- **`avg_leakage`**: The mean prediction residual right before failure.
- **`avg_lyapunov`**: The energy state of the system during failure.
- **`confirmations`**: How many engines have independently developed this exact graph structure.
- **`intervention_success_rate`**: How often breaking this graph prevented failure.

### 2. `MotifFingerprint`
Converts continuous, noisy edge probabilities into a discrete, hashable signature.
- Edges below a specific threshold (e.g., $P < 0.1$) are stripped.
- Remaining structural edges are sorted into a canonical string format and hashed via SHA-256.
- This creates an isomorphic hash allowing CausalNerve to recognize identical failure modes even if edge weights fluctuate.

### 3. `MotifMatcher`
In real-time, the live engine's current adjacency matrix is compared against the entire memory bank.
- We utilize **Jaccard Similarity** over the set of active edges.
- If $J(A, B) > 0.5$, the engine is experiencing a known structural phenomenon.

### 4. `EarlyWarningEngine`
A precognitive monitoring layer that intercepts the graph before catastrophic divergence.
- Normal operation waits for Leakage $L > 0.05$ to trigger an alarm.
- If the `EarlyWarningEngine` detects a motif with high historical transfer confidence, it can trigger an alarm *before* the leakage threshold is crossed.

## Persistence
Stored purely as `motif_memory.json` in `~/.causalnerve/`. No external databases are required, preserving the zero-dependency, edge-deployable nature of CausalNerve.

## Dashboard Integration
The Observatory UI (`causalnerve_observatory.py`) has been upgraded with a **Motif Memory** tab.
When an early warning triggers, it dynamically injects an alert specifying:
- The motif fingerprint
- Jaccard similarity percentage
- Transfer confidence
- Fleet history (which specific engines previously failed in this way)
