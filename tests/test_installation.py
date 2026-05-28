import importlib
import pytest
import sys
import subprocess

def test_core_installation():
    """Verify core package is importable and has correct version."""
    import causalnerve
    assert hasattr(causalnerve, "__version__")
    assert causalnerve.__version__ == "1.0.0"
    
def test_cli_entry_points():
    """Verify that CLI scripts were installed properly."""
    # We just want to check if they are discoverable in the path
    # Using 'where' on Windows, 'which' on Unix
    cmd = "where" if sys.platform.startswith("win") else "which"
    result = subprocess.run([cmd, "causalnerve-demo"], capture_output=True, text=True)
    assert result.returncode == 0, f"causalnerve-demo not found in PATH: {result.stdout}"

def test_optional_dependencies():
    """Verify optional [viz] and [benchmarks] dependencies can be loaded."""
    # This assumes the user ran `pip install -e .[all]`
    for mod in ["fastapi", "dash", "plotly", "scipy", "sklearn"]:
        try:
            importlib.import_module(mod)
        except ImportError:
            pytest.fail(f"Optional dependency '{mod}' is missing. Please run `pip install -e .[all]`.")
