# Reproducibility in CausalNerve Benchmarks

The benchmark suite for `CausalNerve` is designed to be statistically robust and resistant to "cherry-picking" or shortcutting.

## Random Seeds & Statistics
- The suite aggregates results across `N_SEEDS = 10`.
- The `run_all.py` script automatically computes both the **mean** and **standard deviation** for every metric.
- Graphs are generated uniquely for each seed using the standard networkx random algorithms, guaranteeing a wide topological surface area (including chains, trees, Erdős–Rényi, and Scale-Free networks).
- Results presented in our publications utilize these exact scripts. A `benchmark_raw.csv` is emitted prior to aggregation so researchers can compute their own ANOVA or bootstrapped confidence intervals.

## Noise Injection
Every system is simulated under varying SVAR stochastic noise levels:
- `NoiseStd = 0.05` (Clean environment)
- `NoiseStd = 0.15` (Noisy environment)
We strictly evaluate the degradation of detection delay and precision as noise scales. 

## Structural Drift
Rather than testing on static hold-out data, we inject explicit topological drift randomly between time steps $T=200$ and $T=800$.
- `add_edge`, `remove_edge`
- `edge_weight_shift`: sudden strengthening/weakening of causal coefficients.
- `regime_shift`: rapid multi-edge rewiring.
- `cascading_failure`: complete loss of a node's outgoing causal pathways.

## Running the Suite
```bash
python -m benchmarks.run_all
```
This produces:
- `results/benchmark_raw.csv`
- `results/benchmark_table.csv` (aggregated metrics)
