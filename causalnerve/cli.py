import sys
import os

def run_demo():
    """Entry point for causalnerve-demo."""
    # Add root directory to path to allow importing the scripts if installed in dev mode
    sys.path.insert(0, os.getcwd())
    try:
        from causalnerve_demo import main
        main()
    except ImportError:
        print("Error: Could not import causalnerve_demo. Ensure you are running from the root of the project or it is properly installed.")
        sys.exit(1)

def run_observatory():
    """Entry point for causalnerve-observatory."""
    sys.path.insert(0, os.getcwd())
    try:
        from causalnerve_observatory import CausalNerveObservatory
        import uvicorn
        obs = CausalNerveObservatory(port=8765, scenario="fd004")
        obs.start()
    except ImportError as e:
        print(f"Error: Could not import observatory dependencies. Did you install with 'pip install causalnerve[viz]'?\n{e}")
        sys.exit(1)

def run_benchmark():
    """Entry point for causalnerve-benchmark."""
    try:
        from benchmarks.runner import BenchmarkRunner
        runner = BenchmarkRunner(trials=5)
        runner.run()
        runner.generate_report()
    except ImportError as e:
        print(f"Error: Could not import benchmark dependencies. Did you install with 'pip install causalnerve[benchmarks]'?\n{e}")
        sys.exit(1)
