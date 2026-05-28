## Messy Streaming Reality Benchmark Results
## 20 engines, 20 corruption types, seed=42

### Performance Under Sensor Pathologies

| Pathology | FSR (clean) | FSR (corrupted) | Degradation |
|-----------|------------|-----------------|-------------|
| Stuck sensor | 0.19 | 0.22 | +15% |
| Calibration drift | 0.19 | 0.38 | +100% |
| Burst corruption | 0.19 | 0.25 | +31% |
| Duplicated channel | 0.19 | 0.45 | +136% |
| Adversarial combo | 0.19 | 0.62 | +226% |

### Temporal Disorder Tolerance

| Disorder Level | Clean Det.Delay | Corrupted Delay | Increase |
|---------------|-----------------|-----------------|----------|
| Reorder ±2 cyc | 47 cyc | 49 cyc | +2 |
| Reorder ±5 cyc | 47 cyc | 56 cyc | +9 |
| Timestamp ±2σ | 47 cyc | 51 cyc | +4 |
| Async sampling | 47 cyc | 65 cyc | +18 |

### Structural Recovery After Corruption

| Corruption Type | Recovery at +10 | Recovery at +30 | Full? |
|-----------------|-----------------|-----------------|-------|
| Burst (5 cyc) | SHD=2 | SHD=0 | YES |
| Drift (200 cyc) | SHD=4 | SHD=1 | NO (Partial) |
| Override (10 cyc) | SHD=1 | SHD=0 | YES |

### Safe Operating Limits (Empirically Derived)

CausalNerve maintains FSR < 0.30 when:
- Stuck sensor duration: ≤ 50 cycles
- Calibration drift rate: ≤ 0.002 per cycle
- Burst events: ≤ 3 per 100 cycles
- Packet reordering: ≤ ±4 cycles

### Honest Assessment
**Unacceptable Degradation:** Calibration drift and duplicated channels currently trigger false surgeries at unacceptable rates (FSR > 0.40) because the `StructuralAlarmSystem` lacks cross-channel validation for spurious correlation.
**Handled Gracefully:** Burst corruptions and minor temporal jitter (reorder ±2 cycles) are successfully filtered by the Lyapunov gate and dual-world validation.
**Recommendations:** Deployment engineers must configure upstream Kalman filters to remove heavy calibration drift before data enters the CausalNerve observatory.
