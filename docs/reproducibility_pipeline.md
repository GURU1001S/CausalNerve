# CausalNerve Reproducibility Pipeline

A research repository is only as credible as its reproducibility. CausalNerve employs a strict, automated, and cross-platform reproducibility pipeline to ensure all paper figures, benchmark tables, and flagship media can be generated exactly as published, directly from the source code.

## Architecture

The reproducibility system is built on four core pillars:
1. **`reproduce.py`**: The unified, cross-platform entry point for all benchmarks.
2. **GitHub Actions Matrix Testing**: Automated CI pipeline across Ubuntu, Windows, and macOS.
3. **Containerized Environment**: A Docker image guaranteeing OS-level environment isolation.
4. **Deterministic Seed Control**: Strict `PYTHONHASHSEED=42` and explicit PRNG seeding.

## Feature Overview

### 1. Unified Entry Point (`reproduce.py`)
All benchmarking is routed through a single Python script that manages subprocess execution, explicit timeouts, and exact checksum verifications.

*   **Subprocess Isolation**: Prevents namespace pollution between sequentially executed benchmarks.
*   **Timeout Protection**: Kills infinite loops or runaway benchmarks (`subprocess.TimeoutExpired`).
*   **SHA-256 Checksums**: Every benchmark artifact (CSV, GIF, etc.) is strictly verified against an expected SHA-256 checksum, explicitly separating "PASS", "DEGRADED" (completed but mismatched data), and "FAIL".
*   **Fatal Fast-Fails**: If any benchmark fails, the entire pipeline exits with `sys.exit(1)`.

### 2. GitHub Actions (`.github/workflows/reproducibility_check.yml`)
The workflow ensures code does not break across platforms or Python versions.
*   **OS Matrix**: `ubuntu-latest`, `windows-latest`, `macos-latest`.
*   **Python Matrix**: `3.9`, `3.10`, `3.11`.
*   **Smoke Testing**: Runs `pytest tests/ --maxfail=1 -v` before any benchmarks begin.
*   **System Dependencies**: Automatically detects OS and installs FFmpeg via `apt-get`, `brew`, or `choco`.
*   **Caching**: Caches `pip` packages and the `data/` directory to prevent timeout issues on heavy dataset downloads.
*   **Artifact Retention**: Automatically uploads failure logs (`logs/`) and output assets (`results/reproduced/`) directly to the GitHub UI for easy inspection.

### 3. Containerization (`Dockerfile`)
For researchers who want an out-of-the-box experience without modifying their host OS:
*   Pre-installs system dependencies (`ffmpeg`, `libsm6`, `libxext6`).
*   Configures `PYTHONHASHSEED=42` statically.
*   Runs an environment validation check during the build phase to confirm Torch and FFmpeg presence.

## Usage

### Quick Check (15 minutes)
Runs an abbreviated benchmark on 5 specific seeds to verify pipeline integrity.
```bash
python reproduce.py --quick --seeds 42,43,44,45,46
```

### Full Reproduction (2 hours)
Re-runs the entire 50-seed statistical evaluation suite required to generate Paper Table 1 and Figure 3.
```bash
python reproduce.py --all
```

### Targeted Reproduction
Generate only the Flagship self-repair GIF:
```bash
python reproduce.py --benchmark flagship_gif
```

## Troubleshooting & Maintenance
If a benchmark fails with a **DEGRADED** status, it means the script ran to completion but the output SHA-256 hash did not match the paper's canonical hash. This is typically caused by:
1. Floating-point non-determinism (e.g., using different CUDA architectures).
2. Changing `PYTHONHASHSEED` in the environment.
3. Upgrading a core math dependency (like `scipy` or `torch`) to a version that handles tie-breaking differently in isotonic regressions or linear solvers.
