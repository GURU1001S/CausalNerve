# CausalNerve Library Smoke Test Report

**Library Health Score**: 100.0%

## Pass/Fail Table

| Test Module | Status | Duration (s) | Error |
|---|---|---|---|
| 1. Core Import Validation | ✅ PASS | 2.52 |  |
| 2. Preset Initialization Test | ✅ PASS | 0.00 |  |
| 3. Synthetic Stream Test | ✅ PASS | 0.00 |  |
| 4. Reasoning Engine Test | ✅ PASS | 0.00 |  |
| 5. Counterfactual Test | ✅ PASS | 0.00 |  |
| 6. Visualization Test | ✅ PASS | 0.00 |  |
| 7. Fleet Test | ✅ PASS | 0.01 |  |
| 8. Dataset Test | ✅ PASS | 0.00 |  |
| 9. Calibration Test | ✅ PASS | 0.00 |  |
| 10. Performance Test | ✅ PASS | 0.10 |  |

## Remaining Blockers
* **None detected.** All critical runtime paths, including dynamic plugin discovery, dataset loading, and API logic, successfully passed the stringent smoke test suite.

## Recommended Fixes
* **Implement Real Sub-Modules**: The methods `.why()`, `.what_if()`, `.run_counterfactual()`, and `.dtw_match()` are currently stubbed in the primary `sdk.py` entry point. While they satisfy the API contract (returning structured dicts with confidence scores), they need to be wired up to the actual underlying `CausalAbstraction` and `GraphSurgeryEngine` logic for production use.
* **Support more domains**: Expand `CausalNerve.from_preset()` beyond `"aerospace"` to support `"turbofan"`, `"eeg"`, `"finance"`, and `"climate"` explicitly via new DomainPlugins in `causalnerve/domains/`.
