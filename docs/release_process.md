# CausalNerve Release Engineering Protocol

This document defines the strict standard operating procedure for preparing, validating, and releasing a new version of CausalNerve to PyPI and GitHub.

## 1. Pre-Release Checklist
Before attempting to trigger a release, the maintainer must ensure:
- [ ] All CI/CD workflows are passing on `main`.
- [ ] `docs/limitations/FAILURES.md` has been updated with any newly discovered scientific limitations or moved resolved limitations to the "Resolved Issues" block.
- [ ] The `benchmarks/run_all.py` suite executes without regressions in Structural Hamming Distance (SHD) or False Surgery Rate (FSR).
- [ ] No hardcoded absolute paths, large cache files (`.pytest_cache`, `__pycache__`), or bloated `eeg_data` sets have leaked into the repository.

## 2. Version Bump Procedure
We strictly adhere to Semantic Versioning (`MAJOR.MINOR.PATCH`):
1. Navigate to `pyproject.toml` and bump the `version = "X.Y.Z"` variable.
2. If applicable, bump the version string in `causalnerve/__init__.py`.
3. Update `CHANGELOG.md` with a structured list of features, fixes, and scientific enhancements for the new version.

## 3. GitHub Tagging & Automated Release
Once the PR is merged into `main` and the maintainer is ready to push to PyPI:
1. Ensure the remote `main` is perfectly aligned with local.
2. Create an annotated git tag matching the bumped version:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```
3. Pushing the tag automatically triggers the `.github/workflows/publish.yml` GitHub Action.

## 4. Pipeline Execution (`publish.yml`)
The publish workflow will:
1. Spin up a fresh Ubuntu runner.
2. Check out the tagged code.
3. Establish a pristine Python 3.10 environment.
4. Build the `sdist` and `bdist_wheel` formats securely using `python -m build`.
5. Authenticate via OpenID Connect (OIDC) or a PyPI API Token stored securely in GitHub Secrets.
6. Push the compiled wheels and source dists directly to the PyPI index.

## 5. Post-Release Verification
After PyPI confirms receipt of the package, the release engineer must run a public installation smoke test from a completely unlinked terminal:
```bash
pip install causalnerve==<version>
python -c "from causalnerve import CausalNerve; print('PyPI Install OK')"
```
If successful, draft a new Release on the GitHub repository using the contents of the Changelog.
