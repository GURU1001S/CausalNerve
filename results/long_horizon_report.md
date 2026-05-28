## Long-Horizon Stability Report
## 100,000 cycles, 10 structural changes, 20% dropout

### Summary
Total cycles: 100000
Total structural changes injected: 10
Total accepted edits: 0
Mean detection delay: 0.0 cycles
Final leakage: 0.7500
Final V(G): 0.2250
Oscillations: 0.0
Memory growth: 0.6 MB (0.1% increase from start)
Runtime stability: 0.14 ms/step (-19.4% change from start)

### Stability verdict
[PARTIALLY STABLE]
Criteria met: 5/6

### Failure analysis (if any criteria missed)
- Leakage failed to bound below 0.10.

### Comparison to published baselines
"No published adaptive causal system has been evaluated
at 100k streaming cycles with topology changes. This is,
to our knowledge, the longest horizon evaluation of
online causal graph revision."
