# Changelog

## [1.0.5] - 2026-05-28
### Fixed
- Re-added `GraphDiff` class to `causalnerve.memory`.
- Fixed `SyntheticStreamGenerator.with_drift` to accept `n_cycles` parameter.
- Fixed `CausalNerve.why()` to support `target` parameter properly.
- Updated `StructuralReplayEngine.record_revision` and `RevisionRecord` to include `rationale` and `accepted`.
- Fixed `get_graph_diff` implementation in replay engine.
- Ensured `run_intervention` correctly utilizes real `nerve.rollout` mathematical computations instead of hardcoded demo logic.

## [1.0.4] - 2026-05-28
### Fixed
- Fixed missing `retrieve_similar` and broken `predict_transition` in memory bank.
- Restored safe, minimal `StructuralReplayEngine` for observability.
- Ensured graceful degradation of dashboard if replay engine is unavailable.
- Upgraded `observe()` API to launch dynamically.

## [1.0.3] - 2026-05-28
### Fixed
- Fixed a PyPI artifact publishing collision where `causalnerve-observe` was aborted due to pre-existing artifacts for the base package. Both modules have their metadata correctly synced to the `1.0.3` distribution.

## [1.0.2] - 2026-05-28
### Added
- Created isolated automated testing script `scripts/full_external_validation.py` to ensure complete ecosystem stability in fresh Python virtual environments.

### Fixed
- **Memory Packaging:** Added `causalnerve.memory` to the PyPI package containing `StructuralMemoryBank`, `EpisodicMemory`, `RecurrenceEngine`, and `MotifArchive`.
- **Rollout API:** Stabilized the `rollout` signature to `rollout(intervention=None, horizon=50, steps=None)`. The `intervention` argument is no longer required and correctly executes baseline (factual) predictions when omitted.
- **Observability Packaging:** Reconstructed `causalnerve-observe` PyPI deployment. Fixed `observe` import path to execute properly out-of-the-box (`from causalnerve_observe import observe`).

### Changed
- Both `causalnerve` and `causalnerve-observe` version numbers bumped to `1.0.1` for synchronization.
