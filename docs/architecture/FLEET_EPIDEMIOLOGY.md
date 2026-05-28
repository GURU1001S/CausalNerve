# Fleet Structural Epidemiology

## Overview
CausalNerve introduces the world's first **Structural Epidemiology Engine** (`causalnerve/fleet/epidemiology.py`). Rather than analyzing engine failures in isolation, it treats causal failure motifs as "pathogens" that spread across a fleet of identical assets over time, applying public health tracking metrics to industrial hardware.

## Key Components

### 1. `FleetEpidemiologyGraph`
A global, memory-resident directed graph mapping:
- Engines (Nodes)
- Evolving Motifs (Infections)
- Surgical Interventions (Cures)

### 2. `MotifPropagationTracker`
Analyzes how quickly a structural failure motif spreads across the fleet. It computes:
- **Most Contagious Motif**: The causal fingerprint currently infecting the highest number of discrete engines.
- **Prevalence Map**: The penetration percentage of specific structural failures across the fleet.

### 3. `FleetRiskForecaster`
Calculates the **Fleet Stability Index**, a normalized metric indicating the overall causal health of the entire engine fleet based on active motif densities. A dropping index is an early warning indicator for systemic, fleet-wide degradation.

### 4. `TransferLearningLayer`
Enables automated *Causal Immunization*. If Engine-022 successfully survives a structural failure by surgically ablating the $HPT \rightarrow LPC$ edge, and Engine-045 begins developing the same exact isomorphic motif, the Transfer Learning Layer automatically recommends that exact surgery to Engine-045 before the catastrophic failure cascades.

## Dashboard Integration
The Observatory UI's "Fleet Overview" tab has been upgraded to a full **Fleet Epidemiology** dashboard:
- **Fleet Structural Risk**: Real-time readout of the Fleet Stability Index.
- **Most Contagious Motif**: Live identifier of the fastest-spreading failure.
- **Global Failure Clusters**: Shows the top active motifs and their infection counts.
- **Transfer Learning Recommendations**: Proactively suggests surgical graph edits based on the historical survival rates of identical motifs in sister engines.

## Reproducibility
All epidemiological metrics are continuously calculated entirely via deterministic set math on the graph topology, ensuring perfect physical grounding without stochastic ML hallucination.
