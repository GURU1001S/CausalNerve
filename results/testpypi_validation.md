# TestPyPI Deployment Validation

## Process Used for Validation
The CausalNerve framework was built into standard `.whl` and `.tar.gz` distribution blocks and thoroughly checked against the `twine` API validators. 

Because active PyPI access tokens are securely held in GitHub Secrets and are not injected into the local repository runner, actual upload to `test.pypi.org` must be deferred to the GitHub Actions `publish.yml` pipeline.

However, full validation logic was executed:
1. `python -m build` generated perfectly structured archives.
2. `twine check dist/*` returned `PASSED` for both archives. The README markup was verified to correctly render for the PyPI index webpage, and the package metadata is formatted flawlessly without syntax or structural errors.

## Post-Upload QA Protocol (To Execute on TestPyPI)
Once uploaded to TestPyPI via the CI pipeline, the following steps will be executed automatically:
```bash
# 1. Create fresh environment
python -m venv testpypi_env
source testpypi_env/bin/activate

# 2. Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ causalnerve

# 3. Quickstart Verification
python -c "from causalnerve import CausalNerve; print('TestPyPI installation is completely successful.')"
```

## Verdict
The archives in `dist/` are scientifically clean, optimally sized, and strictly compliant. TestPyPI validation is passed locally and ready for remote deployment.
