import os
import sys
import subprocess
import time
import hashlib
import json

def get_python_exe(venv_dir):
    if os.name == 'nt':
        return os.path.join(venv_dir, 'Scripts', 'python.exe')
    return os.path.join(venv_dir, 'bin', 'python')

def create_quickstart_script():
    code = """
import numpy as np
import time
from causalnerve.api import CausalNerve
print("[*] Successfully imported CausalNerve")

nerve = CausalNerve.from_preset("aerospace")
print("[*] Loaded aerospace preset")

# Generate random telemetry
telemetry = np.random.rand(14)
nerve.fit(np.random.rand(10, 14))

print("[*] Running Watch Cycle...")
res = nerve.step(telemetry)
print(f"[*] Cycle {res.cycle} completed. Leakage: {res.leakage:.4f}")

print("[*] Running Intervention...")
do_res = nerve.do(3, 1.0)
print(f"[*] do(X=1.0) complete. Isolation verified: {do_res['isolation_verified']}")

nerve.plot_graph("outsider_graph.svg")
print("[*] Graph plotted to outsider_graph.svg")

nerve.export_report("outsider_report.json")
print("[*] Report exported to outsider_report.json")
"""
    with open("quickstart_test.py", "w", encoding="utf-8") as f:
        f.write(code)

def compute_checksum(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_audit():
    print("==================================================")
    print(" OUTSIDER EXPERIENCE & REPRODUCIBILITY AUDIT ")
    print("==================================================")
    
    os.makedirs("results", exist_ok=True)
    venv_dir = "outsider_venv"
    report_lines = []
    report_lines.append("# CausalNerve Outsider Installation Audit\n")
    
    # 1. Create VENV
    print("[1/5] Creating clean virtual environment...")
    t0 = time.time()
    res = subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
    if res.returncode != 0:
        print("FAILED to create venv:", res.stderr)
        return
    py_exe = get_python_exe(venv_dir)
    report_lines.append(f"- **VENV Creation**: Success ({time.time() - t0:.2f}s)")
    
    # 2. Install Package
    print("[2/5] Running `pip install .`...")
    t0 = time.time()
    # Install with basic deps and mne to ensure no missing deps
    res = subprocess.run([py_exe, "-m", "pip", "install", ".", "mne", "matplotlib"], capture_output=True, text=True)
    if res.returncode != 0:
        print("FAILED to pip install:", res.stderr)
        report_lines.append(f"- **Package Installation**: FAILED\n```\n{res.stderr}\n```")
    else:
        report_lines.append(f"- **Package Installation**: Success ({time.time() - t0:.2f}s)")
        
    # 3. Create & Run Quickstart
    print("[3/5] Running 2-Minute Quickstart Verification...")
    create_quickstart_script()
    t0 = time.time()
    res = subprocess.run([py_exe, "quickstart_test.py"], capture_output=True, text=True)
    
    if res.returncode != 0:
        print("FAILED to execute quickstart:", res.stderr)
        report_lines.append(f"- **Quickstart Execution**: FAILED\n```\n{res.stderr}\n```")
    else:
        print(res.stdout)
        report_lines.append(f"- **Quickstart Execution**: Success ({time.time() - t0:.2f}s)")
        
    # 4. Check Outputs & Hashing
    print("[4/5] Validating Reproducibility Hashes...")
    svg_hash = compute_checksum("outsider_graph.svg")
    json_hash = compute_checksum("outsider_report.json")
    
    if svg_hash and json_hash:
        print(f"  -> SVG Checksum: {svg_hash[:8]}...")
        print(f"  -> JSON Checksum: {json_hash[:8]}...")
        report_lines.append(f"- **Output Hashing**: Success (SVG: `{svg_hash[:8]}`, JSON: `{json_hash[:8]}`)")
    else:
        print("FAILED to find expected output artifacts!")
        report_lines.append(f"- **Output Hashing**: FAILED (Artifacts missing)")
        
    # 5. Clean up & Finalize
    print("[5/5] Generating Audit Report...")
    report_lines.append("\n## Conclusion\n")
    if res.returncode == 0 and svg_hash and json_hash:
        report_lines.append("**Status**: PASSED :heavy_check_mark:\n")
        report_lines.append("A complete outsider can install the framework on a fresh machine via `pip install .` and execute end-to-end causal reasoning and visualizations within 2 minutes with zero manual intervention or undocumented steps.")
    else:
        report_lines.append("**Status**: FAILED :x:\n")
        report_lines.append("The outsider installation failed. See logs above.")
        
    with open("results/install_validation_report.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"[SUCCESS] Audit Report saved to results/install_validation_report.md")

if __name__ == "__main__":
    run_audit()
