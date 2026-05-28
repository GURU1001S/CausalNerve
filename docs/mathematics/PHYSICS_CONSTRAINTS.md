# Physical Constraint Engine

## Overview
Causal AI models traditionally operate entirely on statistical correlations. In complex thermodynamic systems like turbofan engines, this leads to the discovery of statistically robust but physically impossible edges (e.g., downstream sensors "causing" upstream compressor states). 

To solve this, CausalNerve introduces the **Physical Constraint Engine** (`causalnerve/physics/constraint_engine.py`), transforming it into a physically-grounded causal framework.

## Constraint Types

1. **Thermodynamic Directionality**
   * Fluid dynamics dictate that downstream turbine stages cannot causally precede upstream compressor stages unless there is a rigid mechanical spool linking them.
   * *Rule*: Rejects edges from `turbine` or `combustor` to `compressor` unless they map to the specific mechanical spool architecture of the FD004 engine.

2. **Sensor Implausibility**
   * Sensors are observational nodes. They do not inject energy or mass into the system.
   * *Rule*: Rejects any edge originating from a `sensor` node and terminating at a physical component node.

3. **Temperature Propagation Rules**
   * Spontaneous heating violates the Second Law of Thermodynamics.
   * *Rule*: Fluid moving from a low-temperature tier to a high-temperature tier is blocked unless it passes through the `combustor`.

4. **Domain Forbidden Registry**
   * Explicit hardcoded rules for known impossibility paths (e.g., Snsr.B $\rightarrow$ Fan).

## The Mathematics of Rejection
When an edge is evaluated:
1. It begins with a physical plausibility score of `1.0`.
2. As constraints are violated, the score is penalized multiplicatively (e.g., $\times 0.1$ for directionality violation).
3. If the final score drops below `0.3`, the edge is **rejected**.
4. The statistical confidence of the edge is dampened by its physical plausibility ($Conf_{adj} = Conf_{stat} \times Score_{phys}$).

## Dashboard Integration
The Observatory UI has been updated to include a **Physics Violations** panel, providing real-time transparency into how the constraint engine is actively filtering the causal graph. It displays the **Constraint Satisfaction Score** and a live feed of **Rejected Impossible Edges** with natural language explanations.
