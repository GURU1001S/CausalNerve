import subprocess
import venv
import os
import sys

def run():
    print("Creating isolated validation environment...")
    env_dir = "ext_val_env"
    venv.create(env_dir, with_pip=True)
    
    if os.name == 'nt':
        pip_path = os.path.join(env_dir, "Scripts", "pip")
        python_path = os.path.join(env_dir, "Scripts", "python")
    else:
        pip_path = os.path.join(env_dir, "bin", "pip")
        python_path = os.path.join(env_dir, "bin", "python")
        
    print("Installing from local dists to simulate PyPI...")
    # Simulated PyPI install using the local wheels to ensure correctness before actual push
    subprocess.check_call([pip_path, "install", "./dist/causalnerve-1.0.1-py3-none-any.whl"])
    subprocess.check_call([pip_path, "install", "./causalnerve-observe/dist/causalnerve_observe-1.0.1-py3-none-any.whl"])
    
    test_script = """
import sys
try:
    from causalnerve import CausalNerve
    from causalnerve_observe import observe
    from causalnerve.memory import StructuralMemoryBank
    
    nerve = CausalNerve(nodes=5)
    
    # Test Rollout
    ro = nerve.rollout(horizon=10)
    
    # Test Memory
    bank = StructuralMemoryBank()
    
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
"""
    with open("ext_val_test.py", "w") as f:
        f.write(test_script)
        
    print("Running validation script...")
    result = subprocess.run([python_path, "ext_val_test.py"], capture_output=True, text=True)
    
    with open("results/external_validation_report.md", "w") as f:
        f.write("# External Validation Report\n\n")
        f.write("## Execution Logs\n```\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        f.write("\n```\n")
        
        if result.returncode == 0:
            f.write("## Verdict\nSUCCESS: Packages installed cleanly and public APIs are consistent.")
        else:
            f.write("## Verdict\nFAILURE.")

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

if __name__ == "__main__":
    run()
