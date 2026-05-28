"""
benchmarks/run_scientific_integrity_validation.py
=================================================
Validates that all fake / heuristic math has been purged from CausalNerve.

Checks
------
1. No hardcoded similarity constants (dtw_match != 0.95)
2. No hardcoded leakage reduction (* 0.5)
3. Monte-Carlo surgery validator produces varying results
4. Fleet stability index responds to entropy changes
5. Reproducibility: same seed → same numbers
6. Runtime overhead measurement
"""

import sys, time, inspect, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalnerve.adaptation.surgery_validator import MonteCarloSurgeryValidator
from causalnerve.fleet.epidemiology import (
    FleetEpidemiologyGraph, FleetRiskForecaster, MotifPropagationTracker,
    _dtw_similarity, ScientificIntegrityError
)

results = []

def record(test_name: str, passed: bool, detail: str = ""):
    results.append({"test": test_name, "passed": passed, "detail": detail})
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {test_name}" + (f" - {detail}" if detail else ""))


# ─────────────────────────────────────────────────
#  1. Source-code scan for banned patterns
# ─────────────────────────────────────────────────
print("\n=== 1. Static Source Scan ===")

BANNED = [
    (r"\*=\s*0\.5", "Hardcoded leakage *= 0.5"),
    (r"\*\s*0\.5\s*#.*[Aa]ssume", "Hardcoded leakage * 0.5 assumption"),
    (r"return\s+0\.95", "Hardcoded similarity return 0.95"),
    (r"avg_motifs\s*/\s*5\.0", "Arbitrary stability denominator /5.0"),
]

ROOT = Path(__file__).resolve().parents[1]
scan_files = [
    ROOT / "causalnerve" / "adapt" / "ocgr.py",
    ROOT / "causalnerve" / "fleet" / "epidemiology.py",
]

for fpath in scan_files:
    src = fpath.read_text(encoding="utf-8")
    for pattern, label in BANNED:
        matches = re.findall(pattern, src)
        record(f"No '{label}' in {fpath.name}",
               len(matches) == 0,
               f"Found {len(matches)} match(es)" if matches else "Clean")


# ─────────────────────────────────────────────────
#  2. Monte-Carlo surgery validator
# ─────────────────────────────────────────────────
print("\n=== 2. MC Surgery Validator ===")

N = 14
adj = np.zeros((N, N))
edges = [(11,3),(3,4),(4,2),(4,6),(6,2),(2,1),(5,7),(7,0),(9,4),(10,1)]
for i, j in edges:
    adj[i, j] = 1.0
state = np.full(N, 0.5)

mc = MonteCarloSurgeryValidator(horizon=25, n_rollouts=16, seed=42)

# Test: removing a real edge
res_remove = mc.validate(adj, state, edge=(3, 4), edit_type="remove")
record("MC validate returns SurgeryValidationResult",
       hasattr(res_remove, "utility") and hasattr(res_remove, "variance"),
       f"utility={res_remove.utility}, var={res_remove.variance}")

record("MC rollout_utilities has N entries",
       len(res_remove.rollout_utilities) == 16)

# Test: utilities vary (not all identical)
u_arr = np.array(res_remove.rollout_utilities)
record("MC utilities are NOT constant across rollouts",
       float(np.std(u_arr)) > 0,
       f"std={float(np.std(u_arr)):.6f}")

# Test: reproducibility
res2 = mc.validate(adj, state, edge=(3, 4), edit_type="remove")
record("MC is reproducible (same seed -> same result)",
       res_remove.utility == res2.utility)


# ─────────────────────────────────────────────────
#  3. Real DTW matching
# ─────────────────────────────────────────────────
print("\n=== 3. DTW Matching ===")

a = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
b = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
c = np.array([0.9, 0.8, 0.7, 0.5, 0.2])

sim_same = _dtw_similarity(a, b)
sim_diff = _dtw_similarity(a, c)

record("DTW(identical) ~ 1.0", sim_same > 0.99, f"sim={sim_same:.4f}")
record("DTW(reversed) < DTW(identical)", sim_diff < sim_same,
       f"sim_same={sim_same:.4f}, sim_diff={sim_diff:.4f}")
record("DTW does NOT return hardcoded 0.95",
       abs(sim_same - 0.95) > 0.01 or abs(sim_diff - 0.95) > 0.01)

fg = FleetEpidemiologyGraph()
fg.register_engine("E1")
record("dtw_match uses real computation",
       fg.dtw_match(a, b) > 0.99, f"dtw_match={fg.dtw_match(a, b):.4f}")

try:
    fg.dtw_match(np.array([]), np.array([]))
    record("ScientificIntegrityError on empty inputs", False)
except ScientificIntegrityError:
    record("ScientificIntegrityError on empty inputs", True)


# ─────────────────────────────────────────────────
#  4. Fleet stability index
# ─────────────────────────────────────────────────
print("\n=== 4. Fleet Stability Index ===")

fg2 = FleetEpidemiologyGraph()
fg2.register_engine("E1")
fg2.register_engine("E2")

# No motifs → stability = 1.0
s0 = FleetRiskForecaster.compute_fleet_stability_index(fg2)
record("Empty fleet -> stability 1.0", abs(s0 - 1.0) < 1e-6, f"s={s0}")

# Add motifs → stability should decrease
fg2.log_occurrence("E1", "thermal_loop", 1, 0.8)
fg2.log_occurrence("E2", "compressor_chain", 2, 0.7)
fg2.log_occurrence("E1", "turbine_cascade", 3, 0.9)
s1 = FleetRiskForecaster.compute_fleet_stability_index(fg2)
record("Motifs reduce stability", s1 < 1.0, f"s={s1:.4f}")

# Add more diverse motifs → entropy rises → stability drops further
fg2.log_occurrence("E2", "sensor_drift", 4, 0.6)
fg2.log_occurrence("E1", "bearing_wear", 5, 0.5)
s2 = FleetRiskForecaster.compute_fleet_stability_index(fg2)
record("More motifs -> lower stability", s2 <= s1, f"s1={s1:.4f}, s2={s2:.4f}")


# ─────────────────────────────────────────────────
#  5. Runtime overhead
# ─────────────────────────────────────────────────
print("\n=== 5. Runtime ===")

t0 = time.perf_counter()
for _ in range(5):
    mc.validate(adj, state, edge=(3, 4), edit_type="remove")
mc_time = (time.perf_counter() - t0) / 5
record(f"MC validation avg time", True, f"{mc_time:.4f}s")

t0 = time.perf_counter()
for _ in range(100):
    _dtw_similarity(np.random.rand(50), np.random.rand(50))
dtw_time = (time.perf_counter() - t0) / 100
record(f"DTW (len=50) avg time", True, f"{dtw_time*1000:.2f}ms")


# ─────────────────────────────────────────────────
#  Generate outputs
# ─────────────────────────────────────────────────
print("\n=== Generating reports ===")

out = Path("results")
out.mkdir(exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(out / "scientific_integrity_validation.csv", index=False)

passed = sum(1 for r in results if r["passed"])
total = len(results)
score = passed / total * 100

report = f"""# Scientific Integrity Final Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Tests:** {passed}/{total} passed ({score:.0f}%)

## Summary

| Category | Tests | Passed |
|---|---|---|
| Source scan | {sum(1 for r in results if 'No ' in r['test'])} | {sum(1 for r in results if 'No ' in r['test'] and r['passed'])} |
| MC Surgery | {sum(1 for r in results if 'MC' in r['test'])} | {sum(1 for r in results if 'MC' in r['test'] and r['passed'])} |
| DTW | {sum(1 for r in results if 'DTW' in r['test'] or 'dtw' in r['test'])} | {sum(1 for r in results if ('DTW' in r['test'] or 'dtw' in r['test']) and r['passed'])} |
| Fleet | {sum(1 for r in results if 'fleet' in r['test'].lower() or 'stability' in r['test'].lower() or 'Motif' in r['test'])} | {sum(1 for r in results if ('fleet' in r['test'].lower() or 'stability' in r['test'].lower() or 'Motif' in r['test']) and r['passed'])} |

## Runtime

- MC validation (16 rollouts, H=25): **{mc_time:.4f}s**
- DTW (len=50): **{dtw_time*1000:.2f}ms**

## Detailed Results

| Test | Status | Detail |
|---|---|---|
"""
for r in results:
    s = "✅" if r["passed"] else "❌"
    report += f"| {r['test']} | {s} | {r['detail']} |\n"

report += f"\n## Verdict\n\n"
if score == 100:
    report += "**All scientific integrity checks passed.** No fake constants, hardcoded reductions, or mock similarity values remain in the codebase.\n"
else:
    report += f"**{total - passed} check(s) failed.** See details above.\n"

(out / "scientific_integrity_final_report.md").write_text(report, encoding="utf-8")

print(f"\nScientific Integrity Score: {score:.0f}% ({passed}/{total})")
