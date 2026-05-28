# Causal Abstraction Intelligence

## Overview
As causal graphs scale, monitoring raw edges (e.g., `Node 4 -> Node 2: 0.93`) becomes cognitively overwhelming. The **Causal Abstraction Layer** (`causalnerve/reason/causal_abstraction.py`) sits above the raw physics and graph engines. It compresses dense, low-level structural changes into high-level, human-readable semantic macro-states.

## Architecture

### 1. `MotifCompressor`
Translates raw edge matrices into semantic `StructuralEvent` objects.
* Rather than reporting "Edge 0->1 and Edge 1->2 are active", the engine recognizes that nodes 0, 1, and 2 are all compressor stages and outputs a **"Compressor Stage Cascade"** event.
* Groups logical subsystems (Compressors, Combustor, Turbines, Pressures).

### 2. `TemporalPatternMiner`
Maintains a rolling window of graph states to detect non-stationary dynamics.
* By calculating the temporal variance of edge probabilities, it can mathematically detect **Oscillatory Instability**—when the causal structure rapidly toggles back and forth, indicating a system on the edge of chaos.

### 3. `HierarchicalGraphSummarizer`
Translates the list of structural events into a single dominant `MacroCausalState`:
- `NOMINAL`
- `COMBUSTION_INSTABILITY`
- `COMPRESSOR_DEGRADATION`
- `THERMAL_RUNAWAY`
- `OSCILLATORY_INSTABILITY`
- `PRESSURE_IMBALANCE`

### 4. `NarrativeEngine`
Provides deterministic, explainable translation from the mathematical abstraction layer directly into English.
* **Example Output**: *"Instead of tracking 14 low-level edges, abstraction engine detected Combustion Instability. Dominant motif is 'Combustion Dynamics Shift' (Confidence: 0.89). Altered causal flow originating in or affecting combustion stages."*

## Dashboard Integration
This layer is directly hooked into the Observatory UI. The AI Reasoning Narrative has been augmented to display:
- **High-Level Structural State**
- **Dominant Active Motif**
- **Compressed Causal Narrative**

By utilizing deterministic symbolics over the statistical graph outputs, CausalNerve achieves full explainability without relying on hallucination-prone Large Language Models.
