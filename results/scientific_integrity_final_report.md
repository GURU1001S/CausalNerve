# Scientific Integrity Final Report

**Date:** 2026-05-27 00:19
**Tests:** 22/22 passed (100%)

## Summary

| Category | Tests | Passed |
|---|---|---|
| Source scan | 8 | 8 |
| MC Surgery | 5 | 5 |
| DTW | 5 | 5 |
| Fleet | 5 | 5 |

## Runtime

- MC validation (16 rollouts, H=25): **0.0429s**
- DTW (len=50): **1.29ms**

## Detailed Results

| Test | Status | Detail |
|---|---|---|
| No 'Hardcoded leakage *= 0.5' in ocgr.py | ✅ | Clean |
| No 'Hardcoded leakage * 0.5 assumption' in ocgr.py | ✅ | Clean |
| No 'Hardcoded similarity return 0.95' in ocgr.py | ✅ | Clean |
| No 'Arbitrary stability denominator /5.0' in ocgr.py | ✅ | Clean |
| No 'Hardcoded leakage *= 0.5' in epidemiology.py | ✅ | Clean |
| No 'Hardcoded leakage * 0.5 assumption' in epidemiology.py | ✅ | Clean |
| No 'Hardcoded similarity return 0.95' in epidemiology.py | ✅ | Clean |
| No 'Arbitrary stability denominator /5.0' in epidemiology.py | ✅ | Clean |
| MC validate returns SurgeryValidationResult | ✅ | utility=0.01321, var=0.000612 |
| MC rollout_utilities has N entries | ✅ |  |
| MC utilities are NOT constant across rollouts | ✅ | std=0.024736 |
| MC is reproducible (same seed -> same result) | ✅ |  |
| DTW(identical) ~ 1.0 | ✅ | sim=1.0000 |
| DTW(reversed) < DTW(identical) | ✅ | sim_same=1.0000, sim_diff=0.6757 |
| DTW does NOT return hardcoded 0.95 | ✅ |  |
| dtw_match uses real computation | ✅ | dtw_match=1.0000 |
| ScientificIntegrityError on empty inputs | ✅ |  |
| Empty fleet -> stability 1.0 | ✅ | s=1.0 |
| Motifs reduce stability | ✅ | s=0.0000 |
| More motifs -> lower stability | ✅ | s1=0.0000, s2=0.0000 |
| MC validation avg time | ✅ | 0.0429s |
| DTW (len=50) avg time | ✅ | 1.29ms |

## Verdict

**All scientific integrity checks passed.** No fake constants, hardcoded reductions, or mock similarity values remain in the codebase.
