# CausalNerve OnlineCalibrator Fix Report

## 1. Previous Failure
When running the `tests/test_library_smoke.py`, the system threw an `ImportError`:
```
OnlineCalibrator not implemented
```
This was caused by the `OnlineCalibrator` missing entirely from the `causalnerve/adapt/` package.

## 2. Root Cause
The `calibrator.py` module had not been implemented, and consequently, `OnlineCalibrator` was missing from `causalnerve.adapt.__all__`.

## 3. The Fixes Applied
1. **Implementation**: Implemented `causalnerve/adapt/calibrator.py` featuring a rolling-window mechanism (`deque`) to track empirical correctness versus raw output confidence.
2. **Graceful Degradation**: Integrated `sklearn.isotonic.IsotonicRegression` for probability calibration. Crucially, wrapped the import and fitting procedures in `try/except` blocks. If `scikit-learn` is absent, the system safely throws a `UserWarning` and falls back to **identity calibration** (returning the raw confidence score).
3. **Metrics**: Implemented `compute_ece(n_bins=10)` to compute the Expected Calibration Error directly on the rolling window.
4. **Namespace Export**: Added `from .calibrator import OnlineCalibrator` to `causalnerve/adapt/__init__.py`.
5. **Testing**: 
    - The `test_9_calibration` step in the smoke test was updated to import correctly and now passes seamlessly.
    - Created a standalone unit test (`tests/test_calibrator.py`) verifying both the standard isotropic calibration flow and the explicit fallback mechanism.

## 4. Final Status
The `OnlineCalibrator` is now fully operational, tested, securely exported to the top-level API space, and dynamically resilient against missing deployment dependencies.
