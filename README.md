# CausalNerve
CausalNerve: online causal graph adaptation for streaming non-stationary systems.

## Core Capabilities
1. Real-time causal structure revision without retraining (see Section 6 benchmarks).
2. Intervention simulation via Pearl's do-calculus (native) (demonstrated in Turbofan demo).
3. Structural isolation guarantee: do(X) only affects descendants (formalized in PAPER.md).

## Installation
```bash
pip install causalnerve
pip install causalnerve causalnerve-observe  # for dashboard
```

### Upgrade Note
To upgrade to the latest stabilized version:
```bash
pip install --upgrade causalnerve causalnerve-observe
```

## Quickstart
```python
from causalnerve import CausalNerve
from causalnerve.datasets import SyntheticStreamGenerator

nerve = CausalNerve(nodes=6, state_dim=32)
nerve.fit(SyntheticStreamGenerator.stable(n_cycles=200), epochs=20)
nerve.watch(SyntheticStreamGenerator.with_drift(drift_at=100))
```

Expected output:
```text
CausalNerve Summary
────────────────────────────────────────
Cycles processed: 200
Structural alarms: 3
Edits accepted: 1 (edge 4->2, conf=0.71)
Edits rejected: 2
Final leakage: 0.031
Causal equilibrium: reached at cycle 147
────────────────────────────────────────
Run nerve.why(4) for root cause analysis.
Run nerve.what_if({2: 0.5}) to simulate interventions.
```

## What CausalNerve Is Not
- CausalNerve is not a causal discovery algorithm (use PCMCI for that).
- CausalNerve is not a replacement for physics simulations.
- CausalNerve is not validated for safety-critical deployment without domain expert review of its structural revisions.

## Scientific Benchmarks
Evaluated on NASA C-MAPSS FD001 (Engines 81-100).

| Method | SHD ↓ | Det. Delay ↓ | Runtime ↑ | Online? |
|--------|--------|--------------|-----------|---------|
| CausalNerve | 0.0 ± 0.0 | 221.7 ± 60.3 | 83 ms | Yes |
| PCMCI | 105.8 ± 9.3 | N/A (offline) | 4613 ms | No |
| VAR-LiNGAM | 20.0 ± 0.0 | N/A (offline) | 1 ms | No |
| Granger | 158.9 ± 13.7 | N/A (offline) | 780 ms | No |

*Note: CausalNerve is structurally self-repairing (via automated dual-world validation), which incurs an ~80ms overhead compared to static architectures. It trades offline causal discovery accuracy for online adaptability; specifically, CausalNerve loses to PCMCI on Structural Hamming Distance (SHD) for stationary chain graphs.*

## Examples
We provide three definitive demonstrations of the autonomous capabilities:
1. [Real-time Causal Self-Repair in Turbofan Engines](examples/01_turbofan_flagship.ipynb)
2. [Tracking Changing Brain Connectivity During Seizure](examples/02_eeg_seizure_dynamics.ipynb)
3. [Detecting Cascading Failures in Distributed Systems](examples/03_distributed_systems.ipynb)

## Documented Limitations
CausalNerve is bound by constraints in scalability, expected calibration error (ECE) under rapid distribution shifts, and sensitivity to highly correlated sensor noise. For a rigorous audit of system boundaries and known failure modes, please read [FAILURES.md](FAILURES.md).

## Citation
If you use CausalNerve in your research, please cite our technical paper:
```bibtex
@software{causalnerve2026,
  author = {CausalNerve Core Team},
  title = {CausalNerve: Adaptive Structural Dependency Learning},
  year = {2026},
  url = {https://github.com/guru-s/CausalNerve}
}
```

## License
MIT License
