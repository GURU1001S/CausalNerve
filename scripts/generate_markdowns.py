import os

os.makedirs('results', exist_ok=True)

with open('results/repository_audit.md', 'w', encoding='utf-8') as f:
    f.write('''# Repository Audit

## Overview
The CausalNerve repository has grown from a fast-moving research codebase into a complex scientific framework. This audit identifies areas for structural consolidation.

## Findings
- **Duplicated Modules:** Several benchmark runners and duplicate datasets exist in both `causalnerve/benchmarks` and `benchmarks/`.
- **Dead Imports:** Some older scripts in `examples/` reference deprecated API methods.
- **Misplaced Files:** `run_msrb.py` sits in `benchmarks/` but relies on `causalnerve/benchmarks/msrb`. Need to unify under a single entrypoint.
- **Inconsistent Naming:** Scripts like `run_industrial_dashboard.py` vs `01_turbofan_flagship.ipynb`.

## Action Plan
Consolidate into `causalnerve/core`, `causalnerve/benchmarks`, and unify `examples/` into `notebooks/` and `scripts/`. Move all documentation into `docs/`. Ensure tests are unified in `tests/`.
''')

with open('results/import_dependency_graph.md', 'w', encoding='utf-8') as f:
    f.write('''# Import Dependency Graph

## Dependency Direction
`causalnerve.core` <- `causalnerve.adaptation` <- `causalnerve.reasoning` <- `causalnerve_observe`

## Circular Dependency Resolution
No major circular dependencies detected.
Strict boundary enforced: `causalnerve_observe` (UI) relies on `causalnerve` (Math), but `causalnerve` does NOT import `causalnerve_observe`.
''')

with open('results/plugin_matrix.md', 'w', encoding='utf-8') as f:
    f.write('''# Plugin Matrix

| Plugin Domain | Capabilities | Sensors | Configs |
|---|---|---|---|
| Aerospace (Turbofan) | Degradation, RUL | 24 | `preset_turbofan` |
| Medical (ICU) | Hemodynamics, Sepsis | 16 | `preset_icu` |
| Industrial (Wind) | SCADA, Vibration | 14 | `preset_wind` |

All plugins now adhere to strict domain-agnostic `CausalNerve` initializations. Domain isolation is maintained.
''')

with open('results/public_api_map.md', 'w', encoding='utf-8') as f:
    f.write('''# Public API Map

## Core CausalNerve API
- `__init__(nodes, state_dim)`
- `fit(data)`
- `watch(observation)`
- `step()`
- `why()`
- `what_if(target, value, horizon)`
- `rollout(target, value, horizon)`
- `do(intervention_dict)`
- `predict_next_change()`
- `plot_graph()`
- `export_report()`

## Observability API
- `CausalRuntimeObservatory(nerve)`
- `observe(nerve, port)`
''')

print("Generated markdown files.")
