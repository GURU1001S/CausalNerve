# CausalNerve Observatory v1.0

## Release Notes
CausalNerve Observatory v1.0 marks the transition of the framework from an internal engineering prototype to a fully stabilized, scientifically rigorous, and public-ready system. 

### 1. Replay Engine
- **Telemetry Replay (`causalnerve/replay_engine.py`)**: You can now load historical `.jsonl` telemetry files and replay them flawlessly in the dashboard.
- **Data Export**: A new `Export Telemetry` button is available directly in the Observatory UI header, allowing one-click downloads of complete session histories.

### 2. Dashboard Polish & Stability
- **Graceful Loading & Reconnect**: Implemented full `requestAnimationFrame` and async polling wrappers. If the Python backend crashes or stalls, the UI gracefully enters a "Connection Lost" state and automatically recovers when the server reboots.
- **UI Overflow Fixes**: Grid layouts have been adjusted to natively expand based on content rather than rigid absolute heights, eliminating all text bleed bugs (e.g., in the Dual-World Divergence panel).

### 3. Scientific Integrity & Abstraction
- All fake/random dashboard-level placeholders have been replaced. The UI now rigorously renders *only* what is calculated via the physics constraint engine, the temporal motif memory, and the abstraction layer.
- The **Abstraction Layer** deterministically evaluates live matrices and groups them into logical macroscopic phenomena (e.g., *Thermal Runaway*, *Compressor Degradation*).

### 4. Fleet-Scale Structural Epidemiology
- **Motif Tracking**: Real-time cross-engine tracking of the exact moment a specific causal motif emerges.
- **Transfer Learning**: The framework now automatically parses past successful interventions on identical motifs to prescribe high-confidence surgical suggestions for new, un-failed engines.

## Instructions for Replay
To replay a previously exported telemetry session:
```bash
python causalnerve/replay_engine.py <path_to_telemetry.jsonl> --speed 1.0
```
This will automatically spool up a new Dashboard instance and play back the entire graph evolution lifecycle at 50ms intervals.
