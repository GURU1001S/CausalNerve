# Contributing to CausalNerve

We love your input! We want to make contributing to CausalNerve as easy and transparent as possible, whether it's:
- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new physics constraint engines

## Development Setup

1. Fork the repo and clone it locally.
2. Install in editable mode with all dependencies:
   ```bash
   pip install -e .[all]
   ```
3. Run the test suite to ensure your environment is sane:
   ```bash
   pytest tests/ -v
   ```

## Pull Request Process

1. Ensure any new module has a corresponding `tests/test_*.py` file.
2. If you are adding a new baseline to `benchmarks/`, ensure it wraps gracefully so CI/CD doesn't break if the dependency isn't installed globally.
3. Keep the Observatory UI vanilla JS/CSS. Do not introduce heavy frontend frameworks (React, Vue) without core team approval.
4. Update the `README.md` if you are changing core SDK entry points.

## Scientific Integrity

If you are modifying the causal reasoning logic (e.g. `causal_abstraction.py` or `constraint_engine.py`), your PR **must** include a screenshot or export of the `causalnerve-benchmark` report proving that your changes did not degrade the Structural Hamming Distance (SHD) against the VARLiNGAM baseline.
