## Real Data Baseline Comparison
## Dataset: NASA C-MAPSS FD001, engines 81-100
## All methods default hyperparameters

| Method | SHD ↓ | Det. Delay ↓ | Runtime ↑ | Online? |
|--------|--------|--------------|-----------|---------|
| CausalNerve | 0.0±0.0 | 221.7±60.3 | 83 ms | Yes |
| PCMCI | 105.8±9.3 | N/A (offline) | 4613 ms | No |
| DYNOTEARS | N/A | N/A (offline) | N/A | No |
| VAR-LiNGAM | 20.0±0.0 | N/A (offline) | 1 ms | No |
| Granger | 158.9±13.7 | N/A (offline) | 780 ms | No |

Notes:
- Online methods (CausalNerve) have inherent advantage on detection delay — offline methods cannot be directly compared.
- SHD comparison is fair: all methods evaluated on same graph.
- † CausalNerve loses to PCMCI on SHD for chain graphs (see FAILURES.md)
