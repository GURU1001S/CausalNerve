# Long Horizon Causal Optimization

## Overview
CausalNerve has evolved beyond reactive graph repair (identifying a false edge and deleting it) into **Long Horizon Optimization**. The new `LongHorizonEvaluator` provides a mathematically rigorous way to track whether a structural graph surgery actually extended the engine's stable lifetime, or if it merely delayed an inevitable failure.

## The Architecture (`causalnerve/adapt/intervention_memory.py`)

### 1. `InterventionRecord`
When OCGR (Online Causal Graph Revision) executes a structural edit, an `InterventionRecord` is created. It captures:
- The exact edge manipulated.
- The state of the engine immediately before surgery (Leakage $L$, Lyapunov Energy $V$).
- The state immediately after surgery.
- It then opens a tracking horizon.

### 2. `LongHorizonEvaluator`
This engine runs continuously alongside the main monitoring loop. For every active `InterventionRecord`, it wakes up at predefined horizons (e.g., $T+10$, $T+50$, $T+100$, $T+300$ cycles) and logs the ongoing physical state of the engine.
If leakage spikes drastically ($> 1.5\times$ original baseline), the intervention is marked as a **Rollback Occurrence** (a failed repair).

### 3. `InterventionScore`
Calculates the **Delayed Reward** of the intervention. It is a composite metric combining:
- **Structural Stability Gain**: The integral of the Lyapunov energy reduction over the survived horizon.
- **Causal Persistence**: The fraction of the survival duration where leakage remained below the pre-intervention baseline.
- **Longevity**: Total stable cycles gained.

### 4. `SurvivalAnalysis`
Applies actuarial statistics (Kaplan-Meier estimates) to the fleet's historical repairs. It computes a true Survival Curve $P(T > t)$, answering the question: *"If we perform this graph surgery, what is the probability the engine will remain stable for another 100 cycles?"*

## Reproducibility & Auditing
All completed intervention tracking records are automatically exported to a strict CSV file (`intervention_memory.csv`). This ensures zero data loss and allows researchers to import the survival data directly into external statistical software (e.g., R, Pandas) to verify the Kaplan-Meier curves without relying on the dashboard.

## Dashboard Integration
The new **Intervention ROI** tab dynamically tracks:
- Average Delayed Reward score.
- Fleet-wide Rollback Probability.
- Live Kaplan-Meier Survival block rendering.
- A natural-language Audit Layer that translates the math into human-readable engineering logs (e.g., *"Surgical intervention extended stable operation by +120 cycles"*).
