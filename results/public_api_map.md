# Public API Map

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
