# Repository Audit

## Overview
The CausalNerve repository has grown from a fast-moving research codebase into a complex scientific framework. This audit identifies areas for structural consolidation.

## Findings
- **Duplicated Modules:** Several benchmark runners and duplicate datasets exist in both `causalnerve/benchmarks` and `benchmarks/`.
- **Dead Imports:** Some older scripts in `examples/` reference deprecated API methods.
- **Misplaced Files:** `run_msrb.py` sits in `benchmarks/` but relies on `causalnerve/benchmarks/msrb`. Need to unify under a single entrypoint.
- **Inconsistent Naming:** Scripts like `run_industrial_dashboard.py` vs `01_turbofan_flagship.ipynb`.

## Action Plan
Consolidate into `causalnerve/core`, `causalnerve/benchmarks`, and unify `examples/` into `notebooks/` and `scripts/`. Move all documentation into `docs/`. Ensure tests are unified in `tests/`.
