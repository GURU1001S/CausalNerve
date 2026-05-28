# CausalNerve Release Checklist (v1.0.0)

## 1. Dependency Isolation Confirmed
- [x] **Core (`causalnerve`)**: Only installs `torch`, `numpy`, `pandas`, `networkx`. No heavy UI libraries.
- [x] **Viz (`causalnerve[viz]`)**: Installs `fastapi`, `uvicorn`, `plotly`, `dash`.
- [x] **Benchmarks (`causalnerve[benchmarks]`)**: Installs `scipy`, `statsmodels`, `scikit-learn`.

## 2. CLI Entry Points Active
- [x] `causalnerve-demo`: Runs the automated loop.
- [x] `causalnerve-observatory`: Starts the FastAPI telemetry UI on port 8765.
- [x] `causalnerve-benchmark`: Runs the scientific baseline comparator and generates HTML/MD reports.

## 3. Platform & Packaging Checks
- [x] Tested on Windows `win32`.
- [x] Tested on Python `3.9`, `3.10`, `3.11`, `3.12`.
- [x] Semantic versioning implemented (`1.0.0`) in `__init__.py`.
- [x] Clean wheel compilation configuration via `pyproject.toml`.

## 4. Next Steps for Maintainers
1. Run `pip install build`
2. Run `python -m build` to generate the `.whl` and `.tar.gz` artifacts in the `dist/` folder.
3. Validate artifacts using `twine check dist/*`.
4. Upload to PyPI via `twine upload dist/*`.
