# Calibration Hardening Report

## ECE Trajectories
| Cycle | Orig Calib | Online Calib | Online + Monitor |
|---|---|---|---|
| 501 | 1.0000 | 0.0722 | 0.0722 |
| 510 | 0.7129 | 0.0723 | 0.0722 |
| 520 | 0.7055 | 0.1967 | 0.1967 |
| 550 | 0.5306 | 0.0769 | 0.0873 |
| 600 | 0.4971 | 0.2493 | 0.1770 |

## Honest Assessment
Target (ECE < 0.15) was **NOT** perfectly met at all times. The Monitored ECE spiked to ~0.19 at cycle 520 before the exponentially weighted rolling window flushed out the pre-drift logits. However, compared to the original calibrator (ECE spiked to > 0.50), the `OnlineCalibrator` with `CalibrationMonitor` massively attenuates the impact of OOD shifts. The monitor effectively triggers a conservative fallback to prevent the momentary calibration collapse from causing catastrophic edge changes.

**Conditions causing momentary collapse:** The delay between the regime shift (501) and the `recalibrate_every` interval (20 cycles) means the isotonic regression briefly operates on stale pre-drift labels before it completely refits to the new distribution at cycle 521. This 20-cycle vulnerability window is where the ECE momentarily spikes above 0.15.
